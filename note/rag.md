# RAG for Tool Result Filtering

## The Problem

When the agent calls a tool like `sfis_query`, the tool returns a large block of text
(potentially 2000–5000+ characters: 11 structured fields + all parsed SFIS tables).
Right now the entire text is passed as the `Observation` into the LLM's context.
This wastes tokens, can confuse a small local model, and slows down inference.

The goal: give the LLM only what it needs to answer the user's question.

---

## The Vision (user's idea)

Instead of dumping the entire tool result into the LLM at once, process it incrementally:

```
User question
  ↓
LLM reasons → calls tool
  ↓
Tool returns large result
  ↓
RAG processes result chunk by chunk
  for each chunk:
    → Is this chunk relevant to what the user is asking?
      YES → use it to generate a partial answer for that part
      NO  → skip it
  ↓
After processing:
  → Are all parts of the user's question resolved?
      YES → return the composed final answer, stop
      NO  → loop back: LLM reasons again, may call another tool
```

**Key insight:** the agent does not wait for all chunks before answering.
It generates partial answers as relevant chunks are found, and stops
as soon as the full question is resolved — not when all data is exhausted.

---

## Why This Is Better Than Naive Full-Context Injection

| Current approach | RAG approach |
|---|---|
| Pass all 4000 chars to LLM | Pass only 500–800 chars of relevant chunks |
| LLM reads noise before finding the answer | LLM sees only signal |
| Context fills up fast on multi-turn sessions | Context stays lean |
| Fixed result regardless of what user asked | Answer shaped by the actual question |

For a local 9B model with limited context budget, this matters a lot.

---

## Proposed Architecture

### Phase 1 — Batch scoring (simpler, implement first)

```
tool_result (large text)
  ↓
chunk(tool_result)           # split by logical unit (table / search result / paragraph)
  ↓
embed(chunks) + embed(question)    # use ChromaDB's all-MiniLM-L6-v2 already loaded
  ↓
score each chunk by cosine similarity to question
  ↓
keep top-K chunks (until ~1500 chars filled)
  ↓
LLM receives filtered observation → generates answer
```

- One embedding pass, one LLM inference call
- Fast enough for a local model
- Reuses ChromaDB's embedding function (no new dependency)

### Phase 2 — Streaming chunk-by-chunk (user's vision, more advanced)

```
tool_result
  ↓
chunk(tool_result)
  ↓
for each chunk (sorted by relevance score, highest first):
    resolved_parts = LLM("Does this chunk answer part of the question? Which part?")
    if resolved_parts:
        stream partial answer to UI
        mark those sub-questions as done
    if all sub-questions done:
        break   ← stop early, don't process remaining chunks
  ↓
final answer = composed partial answers
```

**Trade-off:** Phase 2 requires multiple small LLM inference calls per tool result.
On a local 9B model at ~40 tok/s, each call adds ~1–3 seconds.
Worth it if the tool result is very large and the question is narrow.

---

## Chunking Strategy

Different tools return different formats — the chunker needs to detect and split correctly:

| Tool | Result format | Split on |
|---|---|---|
| `sfis_query` | `[Table Name]  N rows` sections | `\n[` (each table header) |
| `web_search` | `---` between search results | `\n---\n` |
| `fetch_url` | `---CHUNK BREAK---` already present | `---CHUNK BREAK---` |
| `sfis_2a_defects` | Statistical summary (already small) | No chunking needed |
| `sfis_pvs_query` | Small record list | No chunking needed |
| `read_file` | Plain text | Double newlines `\n\n` |
| Generic | Anything else | Double newlines, fallback to fixed-size |

Minimum chunk size: ~100 chars (avoid splitting single sentences into useless fragments).
Maximum chunk size: ~600 chars (keep each chunk independently meaningful).

---

## Relevance Scoring

### Option A — Cosine similarity (semantic, recommended)
- Embed question and each chunk using the same `all-MiniLM-L6-v2` model already in memory.py
- `score = dot(question_embedding, chunk_embedding)` (both unit-normalized)
- Works well for paraphrase and synonym matching

### Option B — Keyword overlap (fast fallback, no embedding)
- Tokenize question into significant words (remove stopwords)
- Score = count of question words found in chunk / total question words
- Used when embedding model is not available

Hybrid: if embedding fails, fall back to keyword scoring silently.

---

## Hook Point in the Code

In `src/agent.py`, `stream_agent()`:

```python
# Current (line ~387–388):
yield {"type": "tool_result", "name": action, "output": result[:2000]}  # UI shows original
history.append((thought, f"{action}|{action_input}", result))           # LLM sees full result

# After RAG:
yield {"type": "tool_result", "name": action, "output": result[:2000]}  # UI unchanged
filtered = filter_observation(task, result)                              # RAG filters here
history.append((thought, f"{action}|{action_input}", filtered))         # LLM sees filtered
```

The UI always shows the real tool output. Only the LLM's history entry is filtered.

---

## Files to Create / Modify

| File | Change |
|---|---|
| `src/rag.py` | New — `filter_observation()`, `_chunk()`, `_score_chunks()` |
| `src/agent.py` | 2-line change in `stream_agent()` |

No new dependencies. Phase 1 reuses ChromaDB's embedding model.
Phase 2 reuses the LLM instance already in memory.

---

## Open Questions

1. **What counts as "question resolved"?**
   For Phase 2, the agent needs to know when it has enough information.
   Options: keyword check ("found VENDOR: ..."), LLM self-assessment, or
   simply "did the final answer change when we added this chunk?"

2. **Should RAG run on ALL tool calls or only large results?**
   Recommended: only when `len(result) > 2000 chars` and tool is in
   `{sfis_query, fetch_url, web_search, read_file}`.

3. **Should the partial answers be streamed to the UI in Phase 2?**
   Yes — emit a `partial_answer` SSE event type so the user sees progress.

---

## Implementation Order

1. **Phase 1 first** — chunking + batch scoring → filtered observation
   Low risk, measurable improvement, can be done in a day.

2. **Measure** — compare answer quality with/without filter on real SFIS queries.

3. **Phase 2 if needed** — streaming partial answers with early exit.
   Only worth building if Phase 1 still leaves the LLM overwhelmed.
