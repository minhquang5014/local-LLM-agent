# Local LLM Agent

Local ReAct agent (Qwen3-9B, mlx-lm on Apple Silicon) for manufacturing engineers:
SFIS database queries, persistent vector memory, real-time web search, file tools,
and scraping.

> Full architecture, agent flow, and module breakdown live in [CLAUDE.md](CLAUDE.md).
> RAG design: [note/rag.md](note/rag.md). Next-phase research (better RAG, SIP logs,
> CSV): [note/research-rag-logs-csv.md](note/research-rag-logs-csv.md).

---

# Dual-Mac Bridge for Real-Time Web Search

**Status:** ✅ **Built and wired.** The agent on Mac #1 (air-gapped `10.52`) gets
internet by delegating `web_search` / `fetch_url` to a bridge service on Mac #2.

## Why

The agent runs inside the corporate network (`10.52.x.x`), which **cannot reach the
public internet**. SFIS and internal lookups work; external web lookups would fail.
So a second Mac on the outside internet does the searching/scraping and hands results
back — over a private link that has **no route into `10.52`**.

```
   Internet  ──WiFi──►  Mac #2  (web bridge — repo: Web-searching-agent)
                          │
                          │  Thunderbolt Bridge (direct USB-C cable)
                          │  private link 192.168.100.0/24, one-way
                          ▼
   Corporate 10.52  ──►  Mac #1  (this repo — Agent 9B + Web UI + SFIS)
                          ▲
                          │
                   Company users (browser) → http://10.52.x.x:8088
```

Both machines are Mac mini M4 24 GB. Mac #2 doesn't run an LLM — it's a fast
search/scrape proxy.

## Networking

The Mac mini M4 has only **one** built-in Ethernet port, so we don't put both on a
switch. Use **Thunderbolt Bridge** (a direct USB-C cable): macOS auto-creates a
`Thunderbolt Bridge` interface; assign static IPs:

| Machine | Corporate / Internet | Private link to the other Mac |
|---|---|---|
| **Mac #1** (agent, this repo) | Built-in Ethernet → `10.52` LAN | `192.168.100.1` |
| **Mac #2** (bridge) | WiFi → Internet | `192.168.100.2` |

This keeps the internet path and corporate LAN physically separate — Mac #2 has no
route to `10.52`, satisfying the isolation requirement automatically.

> If testing over shared WiFi instead of Thunderbolt, point the agent at Mac #2's
> WiFi IP (see `EXTERNAL_BRIDGE_URL` below).

## How web access works now

The bridge lives in its **own repo** — [github.com/minhquang5014/Web-searching-agent](https://github.com/minhquang5014/Web-searching-agent)
— and runs on Mac #2 as `bridge_api.py` (FastAPI), exposing:

- `POST /search` — search (SearXNG → Brave → DDG) → scrape top pages → trim / adaptive
  RAG → returns `combined_text` ready for LLM context. Body: `{query, num_results, rag}`.
- `POST /scrape` — fetch one public URL → cleaned readable text.
- `GET /health` — liveness (no auth).
- Auth: every request needs the `X-API-Key` header.

On Mac #1 (this repo), `src/tools.py`:

- **`web_search`** → always POSTs to the bridge `/search`, returns the source list +
  scraped content.
- **`fetch_url`** → **splits by host**: public URLs go to the bridge `/scrape`;
  internal/private URLs (`10.52.x.x`, `localhost`, RFC-1918) are fetched **locally**
  — Mac #2 must never reach the `10.52` network. Keeps SFIS scraping working.

Config in `src/config.py` (override via env / `.env`):

```
EXTERNAL_BRIDGE_URL       = http://192.168.100.2:8000   # Mac #2 bridge
EXTERNAL_BRIDGE_API_KEY   = <shared secret, X-API-Key>
EXTERNAL_BRIDGE_TIMEOUT   = 90                            # search+scrape can be slow
```

If the bridge is unreachable, the tool returns a clear "external web bridge offline"
message so the agent can fall back to internal-only reasoning.

## Security rules (hold these)

1. **One-way only:** Mac #1 → Mac #2. Mac #2 never initiates connections to `10.52`.
2. **API key** on every bridge request (`X-API-Key`).
3. **Firewall on Mac #2:** only accept connections from `192.168.100.1` on port `8000`.
4. In production, bind the bridge to the Thunderbolt interface, not the internet side.
5. **Keep Mac #2 awake** (`caffeinate -s` / disable sleep) — a sleeping bridge is the
   main cause of intermittent "can't reach `192.168.100.2`" errors.

## Done / next

- [x] Bridge: search + scrape + adaptive-RAG API on Mac #2 (own repo)
- [x] Thunderbolt link + static IPs + API-key auth
- [x] `web_search` / `fetch_url` wired to the bridge (public vs internal split)
- [x] Inference speedups: token-level Stop, terse thoughts, BM25-only + no memory
      recall on the SFIS path (no embedding load)
- [ ] Next research phase — see [note/research-rag-logs-csv.md](note/research-rag-logs-csv.md):
      parent–child RAG, CSV tool (`read_csv`), SIP log analyzer (`analyze_log` +
      Drain3), MMR de-dup, layered memory.
