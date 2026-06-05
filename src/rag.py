"""
RAG-based tool result filter — Phase 1 (batch scoring).

When a tool returns more text than the LLM needs, filter_observation()
splits the result into logical chunks, scores each chunk against the
user's question by cosine similarity, and returns only the most relevant
chunks (up to MAX_OUTPUT chars total).

The UI always receives the full tool output unchanged.
Only the LLM's history entry is filtered — focused signal, not noise.

See note/rag.md for architecture notes and Phase 2 plan.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Only filter when result exceeds this length
_FILTER_THRESHOLD = 2000

# Maximum chars passed to LLM after filtering
_MAX_OUTPUT = 1500

# RAG stops adding chunks when score drops below this.
# Narrow question ("what is vendor?") → only 1–2 high-scoring chunks pass.
# Broad question ("tell me everything") → more chunks pass until score decays.
# Same value as memory.py _RELEVANCE_THRESHOLD for consistency.
_RELEVANCE_CUTOFF = 0.30

# Chunk size bounds
_MIN_CHUNK = 80
_MAX_CHUNK = 700

# Tools that can return large freeform results worth filtering
_FILTER_TOOLS = {"sfis_query", "fetch_url", "web_search", "read_file"}

# Keyword stopwords for fallback scorer
_STOPWORDS = {
    "a", "an", "the", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "for", "in", "on", "at", "to", "of",
    "and", "or", "but", "not", "with", "this", "that", "what", "how",
    "me", "my", "i", "you", "your", "it", "its", "about", "from", "get",
}


# ------------------------------------------------------------------
# Chunking
# ------------------------------------------------------------------

def _chunk(text: str) -> list[str]:
    """
    Split text into logical chunks based on the format it came from.

    Handles: SFIS full tables, web search results, fetch_url output,
    and generic paragraph text.
    """

    # SFIS traveler: "── Full SFIS Tables ──" separates structured summary from tables
    if "── Full SFIS Tables ──" in text:
        parts = text.split("\n── Full SFIS Tables ──", 1)
        summary = parts[0].strip()          # 11-field structured block — always keep
        tables_section = parts[1] if len(parts) > 1 else ""
        # Each table starts with a "[Table Name]" header on its own line
        table_chunks = re.split(r'\n(?=\[)', tables_section)
        chunks = [summary] + [c.strip() for c in table_chunks if c.strip()]
        return [c for c in chunks if len(c) >= _MIN_CHUNK]

    # Web search results separated by ---
    if "\n---\n" in text:
        chunks = [c.strip() for c in text.split("\n---\n") if c.strip()]
        return [c for c in chunks if len(c) >= _MIN_CHUNK]

    # fetch_url already split with ---CHUNK BREAK---
    if "---CHUNK BREAK---" in text:
        chunks = [c.strip() for c in text.split("---CHUNK BREAK---") if c.strip()]
        return [c for c in chunks if len(c) >= _MIN_CHUNK]

    # Generic text: split on blank lines first
    raw_chunks = [c.strip() for c in re.split(r'\n\n+', text) if c.strip()]

    # Sub-split any chunk that's too large
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


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------

def _keyword_score(question: str, chunk: str) -> float:
    """
    Keyword overlap score (fallback when embeddings unavailable).
    Returns 0.0–1.0: fraction of significant question words found in chunk.
    """
    q_words = {
        w.lower() for w in re.findall(r'\w+', question)
        if w.lower() not in _STOPWORDS and len(w) > 2
    }
    if not q_words:
        return 0.0
    chunk_lower = chunk.lower()
    hits = sum(1 for w in q_words if w in chunk_lower)
    return hits / len(q_words)


def _score_chunks(question: str, chunks: list[str]) -> list[tuple[float, str]]:
    """
    Score each chunk against the question.
    Returns [(score, chunk), ...] sorted by score descending.

    Primary:  cosine similarity via ChromaDB's DefaultEmbeddingFunction
              (all-MiniLM-L6-v2, already loaded for MemoryStore — no extra cost).
    Fallback: keyword overlap score if embedding fails.
    """
    try:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        import numpy as np

        ef = DefaultEmbeddingFunction()
        embeddings = ef([question] + chunks)

        q_emb = np.array(embeddings[0], dtype=float)
        q_norm = np.linalg.norm(q_emb)
        if q_norm == 0:
            raise ValueError("zero question embedding")
        q_emb /= q_norm

        scored: list[tuple[float, str]] = []
        for i, chunk in enumerate(chunks):
            c_emb = np.array(embeddings[i + 1], dtype=float)
            c_norm = np.linalg.norm(c_emb)
            score = float(np.dot(q_emb, c_emb / c_norm)) if c_norm > 0 else 0.0
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    except Exception as e:
        logger.debug("Embedding scoring failed (%s) — using keyword fallback", e)
        scored = [(_keyword_score(question, c), c) for c in chunks]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def filter_observation(
    question: str,
    text: str,
    tool_name: str = "",
    threshold: int = _FILTER_THRESHOLD,
    max_output: int = _MAX_OUTPUT,
    relevance_cutoff: float = _RELEVANCE_CUTOFF,
) -> str:
    """
    Return only the chunks of `text` that are relevant to `question`.

    How the RAG loop works:
    1. Split text into logical chunks (by table / search result / paragraph).
    2. Score every chunk against the question using cosine similarity.
    3. Sort by score descending — most relevant first.
    4. Loop through scored chunks and add each to the context IF:
         a. Its score is >= relevance_cutoff  (RAG gate — stops when relevance decays)
         b. The accumulated context is still within max_output chars
       Stop as soon as either condition fails.
    5. The LLM receives only the accumulated context.

    This means a narrow question ("what is the vendor?") gets 1–2 chunks.
    A broad question gets more, until scores drop below the cutoff.
    Fewer tokens → faster LLM inference.

    The first chunk (structured summary / header) is always kept regardless of
    score — it provides the LLM with essential context to frame its answer.

    Passes through unchanged when:
    - text <= threshold chars (no filtering needed)
    - tool_name not in _FILTER_TOOLS (result is already compact)
    """
    if not text or len(text) <= threshold:
        return text

    if tool_name and tool_name not in _FILTER_TOOLS:
        return text

    chunks = _chunk(text)
    if len(chunks) <= 1:
        return text[:max_output]

    print(f"[RAG] {tool_name or 'tool'}: {len(text)} chars → {len(chunks)} chunks | "
          f"cutoff={relevance_cutoff} | question: {question[:70]!r}")

    scored = _score_chunks(question, chunks)

    # Always include first chunk (structured summary / header)
    first = chunks[0]
    rest = [(s, c) for s, c in scored if c != first]

    selected: list[str] = [first]
    used = len(first)

    for score, chunk in rest:
        # RAG gate: stop when relevance decays below cutoff
        if score < relevance_cutoff:
            print(f"[RAG] score {score:.3f} < cutoff {relevance_cutoff} — stopping loop "
                  f"({len(chunks) - len(selected)} remaining chunks skipped)")
            break

        # Budget gate: stop when context is full
        if used >= max_output:
            break

        remaining = max_output - used
        if len(chunk) <= remaining:
            selected.append(chunk)
            used += len(chunk)
            print(f"[RAG]   + chunk (score={score:.3f}, {len(chunk)} chars)")
        else:
            # Fits partially and score is strong — take a slice
            if remaining > 120:
                selected.append(chunk[:remaining - 1] + "…")
                used = max_output
                print(f"[RAG]   + chunk truncated (score={score:.3f})")
            break

    print(f"[RAG] result: {len(selected)}/{len(chunks)} chunks kept → {used} chars "
          f"(top score={scored[0][0]:.3f})")

    return "\n\n".join(selected)
