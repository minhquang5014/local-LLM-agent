# Research — Better RAG + SIP Log & CSV Processing

Research notes toward three goals: (1) a stronger RAG filter, (2) parsing/filtering
**SIP manufacturing log files**, (3) reading/querying **CSV** files. Design-only —
we implement step by step. Offline constraint: Mac #1 is air-gapped on 10.52, runs
mlx on an M4, so **prefer techniques that need no new model download / no GPU**.

---

## 1. What the referenced sources actually give us

| Source | What it is | Useful to us? |
|---|---|---|
| **agentscope-ai/QwenPaw** | Personal Qwen agent; 3-layer memory (live working context → full verbatim history → distilled knowledge), file/tool guard sandbox | **Yes (memory idea).** The 3-layer memory maps well onto our ChromaDB: keep recent turns verbatim, distil older ones. Also its *file guard* is relevant since we're about to read logs/CSV. |
| **openclaw/openclaw** | Multi-channel gateway (WhatsApp/Telegram/…), `SOUL.md` skill configs, skills system | **Partial.** The "skills/agent as editable config files" echoes our `system_prompt/*.md`. Channels not needed (we have a web UI). |
| **crewAIInc/crewAI** | Role-based multi-agent orchestration (agent + task + tools) | **Later.** Could split work into a "log-analysis" vs "SFIS" vs "web" specialist agent. Heavier; revisit after single-agent RAG is solid. |
| **microsoft/autogen** | Multi-agent conversation + tool-use patterns | **Later.** Same theme as crewAI — orchestration, not core to RAG/files. |
| **llama-index + langgraph** | RAG toolkit: NodeParsers, rerankers, PandasQueryEngine; graph orchestration | **Yes — the core reference.** Directly informs chunking, reranking, and CSV querying below. We borrow the *ideas*, not necessarily the dependency. |

**Takeaway:** the multi-agent frameworks (crewAI/autogen) are tangential to our
current need. The high-value borrows are **llama-index RAG patterns** and
**QwenPaw's layered memory**. Everything below is chosen to run offline with no
new heavy model.

---

## 2. RAG filter improvements (`src/rag.py`)

Current pipeline: `chunk → BM25+vector RRF → relevance cutoff → top chunks`.
Already solid. Candidate upgrades, ranked by ROI vs. cost on our setup:

### 2a. MMR de-duplication (cheap, high value) — general/web content only
After hybrid scoring, pick chunks with **Maximal Marginal Relevance** so we don't
feed three near-identical paragraphs. Balances relevance and diversity. Uses the
embeddings we already computed → **free for web/general**, but **skip for SFIS**
(SFIS is BM25-only now — no embeddings — so use structural dedup by table/row key
instead).

### 2b. Parent–child (hierarchical) retrieval — big win for logs & CSV
Retrieve on **small chunks** (a single log block / CSV row group) but return the
**parent block** (with its header: station, SN, timestamp, column names) so the LLM
gets context around the match. Mirrors llama-index `HierarchicalNodeParser`. We can
implement lightweight: keep a `{chunk → parent_header}` map during `_chunk`.

### 2c. Format-aware chunk sizing
llama-index practical ranges: **400–800 tokens for prose, 80–160 for code/structured**.
Our SFIS/2A/log chunks are structured → keep them small and exact (good for BM25);
web prose → allow larger chunks. Make `_MAX_CHUNK` format-dependent.

### 2d. Semantic chunking (optional, web only)
Split web prose on embedding-similarity breakpoints instead of fixed size. Costs
embeddings → only worth it for large web pages, never for SFIS/logs.

### 2e. Cross-encoder reranker — **deferred** (needs a model)
Best-quality reranking, but requires downloading a reranker model (e.g. bge-reranker)
and running it on the M4. Air-gapped + latency cost → defer unless quality demands it.
MMR (2a) captures most of the diversity benefit for free.

**Plan order:** 2b (parent–child) → 2c (sizing) → 2a (MMR for web) → 2d/2e later.

---

## 3. SIP manufacturing log files (new capability)

Goal: point the agent at a big test/station log and answer "what failed / what
patterns / which error codes", without dumping 50k lines into the LLM.

### 3a. New tool: `analyze_log`
`read_file` already reads raw text; add a log-aware analyzer:
- **Block chunking:** split by timestamp lines / test-step markers / per-SN blocks,
  keep the block header (station, SN, time, PASS/FAIL) with each block (feeds 2b).
- **Severity/keyword filter:** pre-filter to ERROR/FAIL/exceed-limit lines before RAG
  so noise (INFO/DEBUG) never reaches the model.
- **BM25 over blocks:** exact-match on error codes / test names (same engine as SFIS).

### 3b. Drain3 log template mining (pure Python, offline)
[logpai/Drain3](https://github.com/logpai/Drain3) clusters raw log lines into
**templates** and extracts the variable parts. Turns thousands of noisy lines into
"N distinct templates + counts + example vars" — perfect for "summarise this log" /
"which error repeats most". No model, streaming, offline-friendly. Optional dep.

### 3c. Output shape
Return a compact summary: top failing templates, counts, first/last timestamp,
affected SNs — then let RAG surface the specific matching blocks on demand.

---

## 4. CSV processing (new capability)

Goal: load a CSV (test dumps, 2A exports, BOM/vendor sheets) and answer questions
without pasting the whole file.

### 4a. New tool: `read_csv` (pandas, offline)
- **Small CSV:** return schema (columns, dtypes, row count) + `head()` + basic stats
  (`describe`, `value_counts` on key columns). That alone answers most questions.
- **Large CSV:** never dump. Compute aggregates (groupby / value_counts / filters)
  driven by the query. Keep the header row with every row-group chunk for RAG (same
  trick we already use for 2A records).

### 4b. Query-over-CSV — safer than text-to-pandas
llama-index `PandasQueryEngine` lets the LLM *write pandas code* → powerful but
**executes model-generated code** (risk). Prefer a **fixed set of safe operations**
(filter by column=value, groupby+count, top-N, describe) selected by the model via
tool args. Revisit code-exec only in a sandbox.

### 4c. Ties into existing 2A flow
`sfis_2a_defects` already writes Excel for >200 records. `read_csv` closes the loop:
export → load back → ask questions over it.

---

## 5. Proposed roadmap (gradual)

| Phase | Change | Files | Risk |
|---|---|---|---|
| A | Parent–child context in RAG (2b) + format-aware sizing (2c) | `rag.py` | low |
| B | `read_csv` tool — schema + stats + safe aggregates (4a/4b) | `tools.py` | low |
| C | `analyze_log` tool — block chunk + severity filter + BM25 (3a) | `tools.py`, `rag.py` | med |
| D | Drain3 template mining for logs (3b) | new dep, `tools.py` | med (opt dep) |
| E | MMR de-dup for web content (2a) | `rag.py` | low |
| F | Layered memory (QwenPaw-style distillation) | `memory.py` | med |
| G | Multi-agent split (crewAI/autogen style) | larger refactor | high |

**Suggested start:** Phase A (RAG parent–child + sizing) — it directly helps SFIS,
logs, and CSV at once, low risk, no new deps. Then B (CSV) since it's self-contained
and immediately useful for 2A exports.

### Dependency notes (offline-safe)
- `pandas` — likely already present; offline. ✅
- `drain3` — pure Python, no model; must be pip-installed once **with internet**
  (do it on Mac #2 or before air-gapping), then works offline. ⚠️
- Cross-encoder reranker / semantic chunking — need models; **deferred**. ❌ for now.
