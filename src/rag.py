"""
RAG-based tool result filter — hybrid retrieval edition.

Improvements over the baseline cosine-only approach:

1. Query expansion      — expands SFIS/manufacturing abbreviations before scoring
                          so "LC" → "lot number", "DC" → "date code", etc.
                          Helps the scorer find the right chunks even when the
                          user uses shorthand.

2. BM25 keyword scoring — pure Python BM25 (no new dependency) captures exact
                          technical term matches: error codes, station names,
                          component refs (R2251, U7000), SN patterns.
                          Semantic similarity alone misses exact-match signals.

3. Hybrid scoring via   — combines BM25 rank and vector rank using Reciprocal
   Reciprocal Rank       Rank Fusion (RRF).  Neither scorer dominates; each
   Fusion (RRF)          corrects the other's blind spots.

4. Contextual headers   — SFIS table chunks are prefixed with "[From: Table Name]"
                          so the LLM always knows which table a fact came from.

5. Relevance cutoff     — RAG loop stops when scores decay below _RELEVANCE_CUTOFF.
                          Narrow questions get 1–2 chunks; broad questions get more.
                          Fewer tokens → faster local inference.

See note/rag.md for architecture rationale and Phase 2 plan.
"""

from __future__ import annotations

import logging
import math
import re

logger = logging.getLogger(__name__)

# ── Tuneable constants ──────────────────────────────────────────────
_FILTER_THRESHOLD  = 2000    # skip filtering when result is smaller than this
_MAX_OUTPUT        = 1500    # max chars the LLM receives after filtering
_RELEVANCE_CUTOFF  = 0.30    # RRF-normalised score below which chunks are dropped
_MIN_CHUNK         = 80      # discard fragments shorter than this
_MAX_CHUNK         = 700     # sub-split chunks larger than this
_RRF_K             = 60      # RRF smoothing constant (standard value)

# Tools whose results are worth filtering (large / freeform output)
# sfis_2a_defects: statistical summary is small, but inline records (<=200) can be 20k+ chars
_FILTER_TOOLS = {"sfis_query", "fetch_url", "web_search", "read_file", "sfis_2a_defects"}

# ── Query expansion dictionary ──────────────────────────────────────
# Maps user shorthand → expanded terms that appear in SFIS / manufacturing data
_EXPANSIONS: dict[str, str] = {
    # SFIS abbreviations (from sfis_workflow.md)
    "lc":      "lot number lot_no",
    "dc":      "date code date_code",
    "sn":      "serial number serial_number",
    "mo":      "manufacturing order mo_number",
    "pn":      "part number comp_part_no",
    "bom":     "bill of materials hw_bom sw_bom",
    "pvs":     "pvs vendor component traceability",
    # Test station shorthand
    "fct":     "functional circuit test fct",
    "bi":      "burn in burn_in",
    "burn":    "burn in burn_in",
    "wifi":    "wifi wireless test",
    "dfu":     "device firmware update dfu",
    # Failure / quality terms
    "fail":    "failed failure failing test_code error list_of_failing_tests failure_message",
    "error":   "error failed failure test_code list_of_failing_tests failure_message",
    "code":    "error_code test_code list_of_failing_tests failure_message",
    "pass":    "passed passing qa_result",
    "vendor":  "vendor manufacturer tsmc supply",
    "phase":   "phase version_code dvt evt mp",
    "model":   "model hw_bom product family",
    "config":  "config sw_bom configuration",
    "line":    "line virtual_line1 smt production",
    "panel":   "panel smt panel_sn track_no",
    "group":   "group test_group group_name",
    "station": "station test_station",
}

# Standard stopwords to ignore during BM25 tokenisation
_STOPWORDS = {
    "a", "an", "the", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "for", "in", "on", "at", "to", "of",
    "and", "or", "but", "not", "with", "this", "that", "what", "how",
    "me", "my", "i", "you", "your", "it", "its", "about", "from", "get",
    "please", "tell", "show", "give", "find", "check", "query",
}


# ── 1. Query expansion ──────────────────────────────────────────────

def _expand_query(question: str) -> str:
    """
    Append expanded terms for known abbreviations found in the question.
    Original question is preserved; expansions are appended so the scorer
    sees both the original phrasing and the full technical terms.

    Example: "what is the LC for this SN?" →
             "what is the LC for this SN? lot number lot_no serial number serial_number"
    """
    words = re.findall(r'\w+', question.lower())
    extra: list[str] = []
    for word in words:
        if word in _EXPANSIONS:
            extra.append(_EXPANSIONS[word])
    return (question + " " + " ".join(extra)).strip() if extra else question


# ── 2. Chunking with contextual headers ────────────────────────────

def _chunk(text: str) -> list[str]:
    """
    Split text into logical chunks, adding a contextual header to each
    SFIS table chunk so the LLM knows which table the data came from.

    Handles: SFIS traveler (full tables), web search, fetch_url, generic text.
    """

    # 2A defect inline records: "SN: XYZ [1A/2A] cnt=N" blocks
    # Statistical summary (>200 records) already starts with "Total records :"
    # and is small — no chunking needed, falls through to generic handler.
    # Inline records (<=200) have a header then one block per SN.
    if "\nSN: " in text and ("  Group   :" in text or "  Station :" in text):
        lines = text.split("\n")
        header_lines: list[str] = []
        record_blocks: list[str] = []
        current_block: list[str] = []

        in_records = False
        for line in lines:
            if line.startswith("SN: "):
                if current_block:
                    record_blocks.append("\n".join(current_block))
                current_block = [line]
                in_records = True
            elif in_records:
                current_block.append(line)
            else:
                header_lines.append(line)

        if current_block:
            record_blocks.append("\n".join(current_block))

        header = "\n".join(header_lines).strip()
        chunks = ([header] if header else []) + record_blocks
        return [c for c in chunks if len(c) >= _MIN_CHUNK]

    # SFIS traveler: "── Full SFIS Tables ──" divider
    if "── Full SFIS Tables ──" in text:
        parts = text.split("\n── Full SFIS Tables ──", 1)
        summary = parts[0].strip()       # 11-field block — always first chunk
        tables_section = parts[1] if len(parts) > 1 else ""

        # Split into individual tables at each "[Table Name]" header
        raw_tables = re.split(r'\n(?=\[)', tables_section)
        table_chunks: list[str] = []
        for raw in raw_tables:
            raw = raw.strip()
            if not raw or len(raw) < _MIN_CHUNK:
                continue
            # Extract table name for the contextual header
            m = re.match(r'\[([^\]]+)\]', raw)
            table_name = m.group(1) if m else "SFIS Table"
            # Prefix with contextual header so LLM knows the source
            header = f"[From: {table_name}]"
            chunk = f"{header}\n{raw}"
            table_chunks.append(chunk)

        return [summary] + table_chunks

    # Web search results
    if "\n---\n" in text:
        chunks = [c.strip() for c in text.split("\n---\n") if c.strip()]
        return [c for c in chunks if len(c) >= _MIN_CHUNK]

    # fetch_url chunks
    if "---CHUNK BREAK---" in text:
        chunks = [c.strip() for c in text.split("---CHUNK BREAK---") if c.strip()]
        return [c for c in chunks if len(c) >= _MIN_CHUNK]

    # Generic: split on blank lines
    raw_chunks = [c.strip() for c in re.split(r'\n\n+', text) if c.strip()]
    result: list[str] = []
    for chunk in raw_chunks:
        if len(chunk) <= _MAX_CHUNK:
            result.append(chunk)
        else:
            lines = chunk.split("\n")
            current: list[str] = []
            current_len = 0
            for line in lines:
                if current_len + len(line) > _MAX_CHUNK and current:
                    result.append("\n".join(current))
                    current = [line]
                    current_len = len(line)
                else:
                    current.append(line)
                    current_len += len(line) + 1
            if current:
                result.append("\n".join(current))

    return [c for c in result if len(c) >= _MIN_CHUNK]


# ── 3. BM25 scoring ────────────────────────────────────────────────

def _tokenise(text: str) -> list[str]:
    return [
        w.lower() for w in re.findall(r'\w+', text)
        if w.lower() not in _STOPWORDS and len(w) > 1
    ]


def _bm25_scores(query_tokens: list[str], chunks: list[str],
                 k1: float = 1.5, b: float = 0.75) -> list[float]:
    """
    BM25 relevance score for each chunk vs. query_tokens.
    Pure Python — no external dependency.
    """
    tokenised = [_tokenise(c) for c in chunks]
    n = len(chunks)
    avgdl = sum(len(t) for t in tokenised) / n if n else 1

    scores: list[float] = []
    for doc_tokens in tokenised:
        dl = len(doc_tokens)
        score = 0.0
        for term in set(query_tokens):          # unique query terms
            tf = doc_tokens.count(term)
            df = sum(1 for dt in tokenised if term in dt)
            if df == 0:
                continue
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            tf_norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
            score += idf * tf_norm
        scores.append(score)
    return scores


# ── 4. Vector scoring ───────────────────────────────────────────────

def _vector_scores(question: str, chunks: list[str]) -> list[float] | None:
    """
    Cosine similarity via ChromaDB's DefaultEmbeddingFunction (all-MiniLM-L6-v2).
    Returns None on failure so the caller can fall back to BM25-only.
    """
    try:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        import numpy as np

        ef = DefaultEmbeddingFunction()
        embeddings = ef([question] + chunks)

        q_emb = np.array(embeddings[0], dtype=float)
        q_norm = np.linalg.norm(q_emb)
        if q_norm == 0:
            return None
        q_emb /= q_norm

        scores: list[float] = []
        for i in range(len(chunks)):
            c_emb = np.array(embeddings[i + 1], dtype=float)
            c_norm = np.linalg.norm(c_emb)
            scores.append(float(np.dot(q_emb, c_emb / c_norm)) if c_norm > 0 else 0.0)
        return scores

    except Exception as e:
        logger.debug("Vector scoring failed: %s", e)
        return None


# ── 5. Reciprocal Rank Fusion ───────────────────────────────────────

def _rrf_combine(scores_list: list[list[float]], k: int = _RRF_K) -> list[float]:
    """
    Combine multiple ranking signals via Reciprocal Rank Fusion.
    Each score list is converted to a rank ordering; RRF fuses the ranks.
    Returns a combined score per chunk (higher = more relevant).
    """
    n = len(scores_list[0])
    combined = [0.0] * n

    for scores in scores_list:
        # Rank: index 0 = best rank
        ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
        rank_of = [0] * n
        for rank, idx in enumerate(ranked):
            rank_of[idx] = rank
        for i in range(n):
            combined[i] += 1.0 / (k + rank_of[i] + 1)

    # Normalise to 0–1
    max_score = max(combined) if combined else 1.0
    if max_score > 0:
        combined = [s / max_score for s in combined]
    return combined


# ── 6. Hybrid scoring entry point ──────────────────────────────────

def _hybrid_score(question: str, chunks: list[str],
                  bm25_only: bool = False) -> list[tuple[float, str]]:
    """
    Score each chunk using BM25 + vector cosine similarity combined via RRF.
    Falls back to BM25-only if vector scoring is unavailable.

    bm25_only=True skips the embedding step entirely — used for SFIS results,
    which are exact-keyword (serial numbers, U7000, LC/DC, error codes) where
    BM25 is both faster and more precise than semantic similarity.
    Returns [(hybrid_score, chunk), ...] sorted descending.
    """
    expanded = _expand_query(question)
    query_tokens = _tokenise(expanded)

    bm25 = _bm25_scores(query_tokens, chunks)
    vector = None if bm25_only else _vector_scores(expanded, chunks)

    if vector is not None:
        combined = _rrf_combine([bm25, vector])
        method = "BM25+vector RRF"
    else:
        # Normalise BM25 to 0–1 as fallback
        mx = max(bm25) if bm25 else 1.0
        combined = [s / mx if mx > 0 else 0.0 for s in bm25]
        method = "BM25-only"

    scored = list(zip(combined, chunks))
    scored.sort(key=lambda x: x[0], reverse=True)
    logger.debug("Hybrid scoring method: %s", method)
    return scored


# ── Public API ──────────────────────────────────────────────────────

def filter_observation(
    question: str,
    text: str,
    tool_name: str = "",
    threshold: int = _FILTER_THRESHOLD,
    max_output: int = _MAX_OUTPUT,
    relevance_cutoff: float = _RELEVANCE_CUTOFF,
    bm25_only: bool = False,
) -> str:
    """
    Return only the chunks of `text` most relevant to `question`.

    Pipeline:
      expand_query → chunk → BM25+vector hybrid score via RRF →
      relevance-cutoff loop → return top chunks ≤ max_output chars

    The first chunk (structured summary / header) is always kept.
    The UI receives the full unfiltered result; only the LLM history entry
    is filtered.

    Passes through unchanged when:
    - len(text) <= threshold  (small results need no filtering)
    - tool_name not in _FILTER_TOOLS  (result already compact)
    """
    if not text or len(text) <= threshold:
        return text

    if tool_name and tool_name not in _FILTER_TOOLS:
        return text

    chunks = _chunk(text)
    if len(chunks) <= 1:
        return text[:max_output]

    expanded_q = _expand_query(question)
    if expanded_q != question:
        print(f"[RAG] query expanded: {question[:50]!r} → {expanded_q[:80]!r}")

    print(f"[RAG] {tool_name or 'tool'}: {len(text)} chars → {len(chunks)} chunks | "
          f"cutoff={relevance_cutoff} | q: {question[:60]!r}")

    scored = _hybrid_score(question, chunks, bm25_only=bm25_only)

    # First chunk = structured summary / header — always include
    first = chunks[0]
    rest = [(s, c) for s, c in scored if c != first]

    selected: list[str] = [first]
    used = len(first)

    for score, chunk in rest:
        # RAG gate: stop when relevance decays
        if score < relevance_cutoff:
            skipped = len(chunks) - len(selected)
            print(f"[RAG] score {score:.3f} < cutoff → stop ({skipped} chunks skipped)")
            break

        if used >= max_output:
            break

        remaining = max_output - used
        if len(chunk) <= remaining:
            selected.append(chunk)
            used += len(chunk)
            print(f"[RAG]   + {chunk[:40]!r}… (score={score:.3f}, {len(chunk)} chars)")
        elif remaining > 120:
            selected.append(chunk[:remaining - 1] + "…")
            used = max_output
            print(f"[RAG]   + truncated (score={score:.3f})")
            break

    print(f"[RAG] ✓ {len(selected)}/{len(chunks)} chunks → {used} chars "
          f"(top={scored[0][0]:.3f})")

    return "\n\n".join(selected)
