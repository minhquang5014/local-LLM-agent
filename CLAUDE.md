# Local LLM Agent

Local ReAct agent running the Qwen3-9B model (safetensors format). Capabilities: web search, persistent vector memory, SFIS manufacturing database query, file tools, and web scraping.

**Primary runtime:** Mac Mini M4 24GB — uses mlx-lm for fast native inference (~40 tok/s).
**Secondary:** Asus Laptop with NVIDIA GPU — falls back to HuggingFace transformers (CUDA).

---

## Architecture

- `src/agent.py` — LangGraph ReAct loop (think → act → respond) + sfis guard + stop_event
- `src/inference.py` — LLM backend: Ollama → mlx-lm → transformers (auto-fallback), Qwen3 chat template
- `src/tools.py` — 9 LangChain tools (web_search, fetch_url, read_file, list_dir, memory_store, memory_recall, sfis_query, sfis_2a_defects, sfis_pvs_query)
- `src/memory.py` — ChromaDB vector store (relevance threshold 0.30, near-duplicate check 0.90, timestamps)
- `src/sfis.py` — Headless SFIS client (http://10.52.1.9): traveler, 2A defects, PVS traceability
- `src/rag.py` — RAG filter: BM25 + vector hybrid scoring via RRF, query expansion, relevance cutoff
- `server.py` — FastAPI + SSE streaming + GZip middleware (port 8088), stop endpoint
- `static/index.html` — Chat UI with dark/light theme toggle, stop button
- `system_prompt/` — System prompt files: main.md, sfis_workflow.md, response_format.md

---

## Full Agent Flow

```mermaid
flowchart TD
    A([User types a message]) --> B

    subgraph UI ["Browser UI  —  static/index.html  ·  dark/light theme"]
        B[Send via SSE\nPOST /api/chat/stream]
        STOP_BTN[Stop button pressed\nPOST /api/chat/stop]
    end

    B --> C

    subgraph SERVER ["Web Server  —  server.py  ·  GZip middleware"]
        C[FastAPI receives request\nattaches session history\ncreates stop_event per session]
        STOP_EP[/api/chat/stop\nsets stop_event for session]
    end

    STOP_BTN --> STOP_EP

    C --> D

    subgraph AGENT ["ReAct Loop  —  src/agent.py  ·  max 10 iterations"]
        D[Build Prompt] --> D0
        D0[Load system_prompt/*.md\nmain · sfis_workflow · response_format] --> D1
        D1[Inject chat history\nlast 4 turns] --> D2
        D2[Auto memory recall\nChromaDB semantic search\ninjected as Relevant Memories] --> STOP_CHK

        STOP_CHK{stop_event\nset?}
        STOP_CHK -->|Yes| STOPPED[Emit stopped event\nabort loop]
        STOP_CHK -->|No| E

        E[LLM Inference\nsrc/inference.py\nQwen3 chat template applied] --> E1

        subgraph LLM ["LLM Backend  —  auto-selected"]
            E1{Is Ollama running?}
            E1 -->|Yes| E2[Ollama\nqwen2.5:7b]
            E1 -->|No| E3{Apple Silicon?}
            E3 -->|Yes| E4[mlx-lm\n~40 tok/s on M4]
            E3 -->|No| E5[HuggingFace transformers\nCUDA / CPU]
        end

        E2 & E4 & E5 --> F

        F[Parse LLM output\nstrip think blocks\ncut hallucinated Observations] --> G{Output type?}

        G -->|Thought + Action| GUARD
        G -->|Final Answer| SFIS_GUARD

        GUARD{sfis_query with\nempty input?}
        GUARD -->|Yes — extract SN\nfrom task| H
        GUARD -->|No| H

        SFIS_GUARD{SN task but\nsfis_query never called?}
        SFIS_GUARD -->|Yes — force call| H
        SFIS_GUARD -->|No| DONE[Exit loop]

        subgraph TOOLS ["Tools  —  src/tools.py  ·  9 tools"]
            H --> H1{Which tool?}

            H1 -->|web_search| WS
            subgraph WS_BOX ["Web Search  —  auto-fallback chain"]
                WS{SearXNG\nrunning?}
                WS -->|Yes| WS1[SearXNG\nlocalhost:8080\nno rate limits]
                WS -->|No| WS2{BRAVE_SEARCH\n_API_KEY set?}
                WS2 -->|Yes| WS3[Brave Search API\n~1000 req/mo free]
                WS2 -->|No| WS4[DuckDuckGo\nddgs · may rate-limit]
            end

            H1 -->|sfis_query| SF
            subgraph SF_BOX ["SFIS Traveler  —  src/sfis.py"]
                SF[Login to\nhttp://10.52.1.9] --> SF1[Query traveler\nfor serial number]
                SF1 --> SF2[Extract structured summary\nPHASE · MODEL · CONFIG · LINE\nPANEL SN · FAILED DATE · GROUP NAME\nFAILURE MESSAGE · LIST OF FAILING TESTS\nLAB IN TIME]
                SF2 --> SF3[Append full table dump\n── Full SFIS Tables ──\nall rows compact KEY=VALUE]
                SF3 --> SF5[Return result\nauto-save to ChromaDB memory]
            end

            H1 -->|sfis_2a_defects| SF2A
            subgraph SF2A_BOX ["SFIS 2A Defects"]
                SF2A[Query defect records\nby date range] --> SF2A1{Records > 200?}
                SF2A1 -->|Yes| SF2A2[Statistical summary\n+ Excel export]
                SF2A1 -->|No| SF2A3[Inline records\nper SN]
            end

            H1 -->|sfis_pvs_query| SFPVS
            subgraph SFPVS_BOX ["SFIS PVS Traceability"]
                SFPVS[Query component vendor\nLOT NO · DATE CODE\nby SN + location]
            end

            H1 -->|memory_recall| MR
            subgraph MEM_BOX ["Memory  —  src/memory.py · ChromaDB"]
                MR[Semantic vector search\nrelevance threshold 0.30\nreturns top-5 with timestamps]
                MS[Save fact\nnear-duplicate check 0.90\nadd Unix timestamp]
            end

            H1 -->|memory_store| MS
            H1 -->|fetch_url| FU[HTTP GET\nBeautifulSoup extraction\nchunked for large pages]
            H1 -->|read_file| FS[Read local file\nmax 1 MB]
            H1 -->|list_dir| FS
        end

        WS1 & WS3 & WS4 --> RAW
        SF5 --> RAW
        SF2A2 & SF2A3 --> RAW
        SFPVS --> RAW
        MR & MS --> RAW
        FU --> RAW
        FS --> RAW

        RAW[Raw tool result\nemitted to UI as tool_result SSE] --> RAG

        subgraph RAG_BOX ["RAG Filter  —  src/rag.py"]
            RAG[filter_observation\nresult > 2000 chars?]
            RAG -->|No — pass through| OBS
            RAG -->|Yes| RAG1[Query expansion\nLC→lot_no DC→date_code\nerror→list_of_failing_tests etc]
            RAG1 --> RAG2[Chunk by format\nSFIS tables · web results\n2A records · generic]
            RAG2 --> RAG3[Hybrid score\nBM25 + vector cosine\ncombined via RRF]
            RAG3 --> RAG4[Relevance cutoff loop\ndrop chunks score < 0.30\nmax 1500 chars output]
            RAG4 --> OBS
        end

        OBS[Filtered observation\nadded to LLM history] --> ITER{Max iterations\nhit?}
        ITER -->|No — loop back| D
        ITER -->|Yes| FALLBACK[Return fallback\nmessage]
    end

    DONE --> SAV[Save Task+Answer\nto ChromaDB memory\nskip if SFIS tool was used]
    SAV --> STREAM
    FALLBACK --> STREAM
    STOPPED --> STREAM

    subgraph OUT ["Output  —  SSE events"]
        STREAM[Stream events to browser\nthought · tool_call · tool_result\nanswer · stopped · error · done]
    end

    STREAM --> END([User sees response])
```

### Event stream from agent to UI

Each step in the loop emits an SSE event the browser renders in real time:

| Event | When emitted | UI display |
|-------|-------------|------------|
| `start` | Loop begins | — |
| `thought` | LLM produces a Thought | Italicised thought block |
| `tool_call` | Tool about to run | Collapsible tool card with icon |
| `tool_result` | Tool returned | Result inside the card (full, unfiltered) |
| `answer` | Final Answer produced | Markdown-rendered answer |
| `stopped` | User pressed Stop button | "Generation stopped" notice |
| `error` | Exception thrown | Error message |
| `done` | Loop finished | — |

---

## Changes Made (2026-05-16)

### src/sfis.py — Structured field extraction
**Problem:** Original code dumped all SFIS tables as raw key-value pairs — verbose, noisy, and hard for the LLM to reason about.

**Fix:** Rewrote `_query_traveler` to extract exactly the fields engineers care about, matching the logic from `test/get_fa_data_on_sfis 6.py`:
- `PHASE`, `MODEL`, `CONFIG` from "Work Order / Model Data"
- `LINE`, `PANEL SN` from "SN Detail Data"
- `SN SEQ IN PANEL` from "Wip Tracking Data"
- `FAILED DATE`, `GROUP NAME`, `LIST OF FAILING TESTS` from "SN Repair Data" (filtered to failure stations: burn_in, fct, cell, wifi, etc.)
- `LAB IN TIME` from "Laboratory In/Out"
- `FAILURE MESSAGE` + refined test list from "Bobcat Data" (matched by FAAP station + stop time)

Output is now a clean, ordered list of fields instead of a raw table dump.

### src/tools.py — Robust web search
**Problem:** `from ddgs import DDGS` was hard-coded — crashes if the older `duckduckgo_search` package is installed instead. No retry on rate-limit errors.

**Fix:**
- Import tries `duckduckgo_search` first, then `ddgs`, then returns a clear install message if neither is found
- Added 3-attempt retry with exponential backoff (1s → 2s → 4s) to handle DuckDuckGo rate limiting

### src/memory.py — Better retrieval quality
**Problem:** Memory search returned all results regardless of relevance score, accumulated duplicates, no way to know when memories were stored.

**Fix:**
- **Relevance threshold (0.30):** results below this cosine-similarity score are filtered out — stops irrelevant memories polluting the context
- **Near-duplicate detection (0.90):** before saving, checks if a very similar memory already exists — prevents the store from growing with repeated facts
- **Timestamps:** every memory is stored with a Unix timestamp; `search_text` shows "5m ago / 2h ago / 3d ago" next to each result
- **`recent(k)` method:** returns the k most recently added memories sorted by time

### src/agent.py — Smarter prompting
- Updated system prompt (removed "Mac Mini M4" reference, added guidance on memory use and concise storage)
- **Auto memory recall:** on the first step of every task, relevant memories are automatically searched and injected into the prompt as "Relevant Memories" — the agent sees prior findings without having to call `memory_recall` manually
- Added `sfis_query: "🏭"` to tool icons for the web UI

### requirements.txt
- Replaced `duckduckgo-search>=6.0.0` → `ddgs>=7.0.0` (current package name)
- Replaced `llama-cpp-python` (unused, Apple Metal comment was misleading) → `mlx-lm>=0.19.0`
- Added notes explaining the two-backend setup

---

## Web Search Engine Options

Current tool uses DuckDuckGo (ddgs), which can be rate-limited and blocked. Fully local alternatives to consider:

| Option | Type | Pros | Cons | Best For |
|--------|------|------|------|----------|
| **SearXNG** | Self-hosted metasearch | Best quality, JSON API, aggregates multiple engines | Needs Docker/server running locally | Drop-in DuckDuckGo replacement |
| **Crawl4AI** | Local scraper | Full page content, JS support | Slower, no keyword search | Enhancing `fetch_url` |
| **Playwright** | Browser automation | Most powerful, handles any site | Heavy, needs more code | Advanced scraping |
| **Jina Reader** | URL-to-markdown | Very clean text output | Needs URL first, no search | Replacing `fetch_url` |

**Recommendation:** Replace DuckDuckGo with **SearXNG** for web search, and optionally replace `fetch_url` with **Crawl4AI** or **Jina Reader** for cleaner page content. See notes below.

---

## Web Search Backends

The agent auto-detects the best available search backend. Priority order:

```
SearXNG (local) → Brave Search API → DuckDuckGo
```

Each result block is labelled with the source used (e.g. `[Search via Brave Search]`).

---

### Option 1 — Brave Search API (Recommended for Mac, no Docker needed)

Free tier: ~1000 requests/month (enough for daily dev use).

**Setup (5 minutes):**
1. Register at https://api-dashboard.search.brave.com
2. Create an API key under "API Keys"
3. Add to your environment:
```bash
export BRAVE_SEARCH_API_KEY=your_key_here
```
Or add to a `.env` file in the project root (loaded automatically via python-dotenv):
```
BRAVE_SEARCH_API_KEY=your_key_here
```

No installation beyond the existing `requests` package needed.

---

### Option 2 — SearXNG (Fully Local, No API Key)

Best long-term option — no quotas, no external dependencies, aggregates Google + Bing + DDG + Brave.

#### Step 1 — Copy files to USB (done on Windows machine with internet)

Two files are already saved on the USB drive (E:):

| File | Size | Purpose |
|------|------|---------|
| `OrbStack-arm64.dmg` | 404 MB | Container runtime for Apple Silicon Mac |
| `searxng-image.tar` | 92 MB | SearXNG Docker image (offline, no internet needed) |

To re-download if needed (run on Windows with Docker Desktop open):
```powershell
# OrbStack installer
Invoke-WebRequest -Uri "https://orbstack.dev/download/stable/latest/arm64" -OutFile "E:\OrbStack-arm64.dmg"

# SearXNG image
docker pull ghcr.io/searxng/searxng:latest
docker save ghcr.io/searxng/searxng:latest -o "E:\searxng-image.tar"
```

#### Step 2 — Install OrbStack on Mac (from USB, no internet needed)

1. Plug in USB — note the volume name shown in Finder under Locations (e.g. `MY_USB`)
2. Double-click `OrbStack-arm64.dmg` on the USB
3. Drag OrbStack to Applications
4. Launch OrbStack once — it installs its components and starts automatically

#### Step 3 — Load SearXNG image and start

```bash
# Load image from USB (no internet needed)
docker load -i /Volumes/YOUR_USB_NAME/searxng-image.tar

# Start SearXNG
cd /path/to/local-LLM-agent
docker compose up -d        # starts at http://localhost:8080
docker compose down         # stop when not needed
```

Replace `YOUR_USB_NAME` with the actual USB volume name shown in Finder.

#### Step 4 — Verify

```bash
curl "http://localhost:8080/search?q=test&format=json"
```

Should return JSON. If you see `403 Forbidden`, wait 10 seconds and retry — container may still be starting.

**OrbStack starts automatically on Mac login** — after initial setup, just run `docker compose up -d` each session.

Config files: `docker-compose.yml`, `searxng/settings.yml` (JSON API + rate limiting disabled already configured).

---

### Option 3 — DuckDuckGo (Built-in Fallback)

Requires `pip install ddgs`. Works out of the box but can be rate-limited under heavy use.

---

## Known Issues / TODO

- Memory embedding uses ChromaDB's default model (all-MiniLM-L6-v2), downloaded on first run
- Session history is in-memory only — lost on server restart
- No token budget management — very long prompts may overflow Qwen context window
