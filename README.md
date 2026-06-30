# Local LLM Agent

Local ReAct agent (Qwen3-9B, mlx-lm on Apple Silicon) for manufacturing engineers:
SFIS database queries, persistent vector memory, web search, file tools, and scraping.

> Full architecture, agent flow, and module breakdown live in [CLAUDE.md](CLAUDE.md).
> RAG design notes live in [note/rag.md](note/rag.md).

---

# PLAN — Dual-Mac Bridge for Real-Time Web Search

**Status:** 📝 Design note only — not yet implemented. We will build this step by step.

## Problem

The agent runs on a machine inside the corporate network (`10.52.x.x`). That network
**cannot reach the public internet**, so `web_search` and `fetch_url` currently fail
for any real-time / external lookup. SFIS and internal queries work fine; external
web lookups do not.

## Goal

Keep the agent + web UI on the internal `10.52` machine (where the users and SFIS are),
but **offload internet access to a second Mac** that is on the outside network. The
internal agent calls the external Mac only when it needs to search or scrape the web.

```
   Internet  ──WiFi/Ethernet──►  Mac #2  (External "web bridge")
                                   │
                                   │  Thunderbolt Bridge (direct USB-C cable)
                                   │  private link 192.168.100.0/24, one-way
                                   ▼
   Corporate 10.52  ──Ethernet──►  Mac #1  (Agent 9B + Web UI + SFIS)
                                   ▲
                                   │
                            Company users (browser) → http://10.52.x.x:8088
```

## Verdict: feasible ✅

Both Macs are M4 24 GB — plenty for a 4-bit 9B model on Mac #1 and a lightweight
scrape/search server on Mac #2 (Mac #2 doesn't even need to run an LLM).

---

## Networking — how the two machines talk

**Key fact:** the Mac mini M4 has only **one** built-in Ethernet port. So we do *not*
plug both Macs into one switch. Instead:

| Machine | Interface 1 | Interface 2 |
|---|---|---|
| **Mac #1** (internal) | Built-in Ethernet → corporate `10.52` LAN | Thunderbolt Bridge → Mac #2 |
| **Mac #2** (external) | WiFi (or its own Ethernet/router) → Internet | Thunderbolt Bridge → Mac #1 |

**Thunderbolt Bridge** = connect the two Macs directly with a USB-C / Thunderbolt
cable. macOS auto-creates a `Thunderbolt Bridge` network interface. Assign static IPs:

- Mac #1 bridge interface: `192.168.100.1`
- Mac #2 bridge interface: `192.168.100.2`

This gives a private, fast (10 Gbps+), point-to-point link that is **physically separate**
from both the corporate LAN and the internet. Mac #2 has no route to `10.52`, which
satisfies the security requirement automatically.

> Alternative if Thunderbolt isn't usable: a USB-C→Ethernet adapter on one Mac + a
> crossover/normal Ethernet cable directly between the two, same static-IP idea. Avoid
> putting both on the corporate switch — that would expose Mac #2's internet path to `10.52`.

---

## How it maps onto this codebase

Today `src/tools.py` runs the search/scrape locally on Mac #1:

- `_web_search()` → SearXNG → Brave → DuckDuckGo  (all need internet → all fail on 10.52)
- `_fetch_url()` → direct HTTP GET + BeautifulSoup  (needs internet → fails on 10.52)

The change is to **route those two tools through the bridge** when a bridge URL is configured:

```
Mac #1 (this repo)                          Mac #2 (new tiny service)
──────────────────                          ─────────────────────────
_web_search(query)                          bridge_server.py (FastAPI)
   └─ if EXTERNAL_BRIDGE_URL set:             POST /search  {query}
        POST http://192.168.100.2:9090  ───►   → SearXNG / Brave / DDG (has internet)
        with X-API-Key header                   → return formatted results
   └─ else: current local behaviour         POST /scrape  {url}
                                                → requests + BeautifulSoup (+ Playwright later)
_fetch_url(url)                                 → return cleaned text
   └─ same routing
```

Nothing about the agent loop, RAG, SFIS, memory, or the web UI changes — only the two
network-bound tools learn to delegate to the bridge.

---

## Security rules (must hold)

1. **One-way only:** Mac #1 → Mac #2. Mac #2 never initiates connections to `10.52`
   and has no network route to it (guaranteed by the isolated Thunderbolt link).
2. **API key** on every bridge request (`X-API-Key`), checked by `bridge_server.py`.
3. **Firewall on Mac #2:** only accept connections from `192.168.100.1` on port `9090`.
4. Bridge server binds to the bridge interface only (`192.168.100.2`), **not** `0.0.0.0`
   on the internet-facing side.

---

## Implementation steps (do one at a time)

- [ ] **Step 0 — Physical link.** Connect the USB-C cable, enable Thunderbolt Bridge on
      both Macs, set static IPs (`.1` / `.2`), verify `ping 192.168.100.2` from Mac #1.
- [ ] **Step 1 — Bridge server skeleton.** New `bridge/bridge_server.py` (FastAPI) on
      Mac #2 with `/health`, API-key middleware. Run on `192.168.100.2:9090`.
- [ ] **Step 2 — `/search` endpoint.** Move the SearXNG/Brave/DDG fallback logic into the
      bridge; return the same formatted string `_format_results()` produces today.
- [ ] **Step 3 — `/scrape` endpoint.** Move `_extract_text()` + chunking logic; return
      cleaned page text.
- [ ] **Step 4 — Client routing in `tools.py`.** Add `EXTERNAL_BRIDGE_URL` +
      `EXTERNAL_BRIDGE_API_KEY` to `src/config.py`. In `_web_search`/`_fetch_url`, if the
      bridge URL is set, POST to it; otherwise keep the current local path (so the laptop
      dev setup still works).
- [ ] **Step 5 — Failure handling.** Clear message to the agent when the bridge is
      unreachable ("external web bridge offline") so it can fall back to internal-only.
- [ ] **Step 6 — Harden.** API key, firewall rule on Mac #2, bind addresses, basic
      request logging.
- [ ] **Step 7 — (optional) Playwright** on Mac #2 for JS-heavy sites, behind the same
      `/scrape` endpoint.

---

## Open questions / decisions

- **Which search backend on Mac #2?** SearXNG (local container, no quotas) vs Brave API
  (simplest, ~1000 req/mo free). Recommend SearXNG since Mac #2 is dedicated to this.
- **Run an LLM on Mac #2?** Not needed for v1 — it's a dumb search/scrape proxy. Keep the
  reasoning on Mac #1. Revisit only if we want the external side to summarise before returning.
- **Caching:** should the bridge cache recent searches to cut latency / external calls?
  (LAN round-trip is <1 ms; the real cost is the upstream search engine.)
