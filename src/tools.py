"""
LangChain-compatible tools for the agent.

Available tools:
  - web_search      — DuckDuckGo search returning summaries
  - fetch_url       — Download and clean a web page
  - read_file       — Read a local file
  - list_dir        — List files in a directory
  - memory_store    — Save a fact to vector memory
  - memory_recall   — Retrieve relevant memories
  - sfis_query      — Query the internal SFIS manufacturing system
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from langchain_core.tools import Tool

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# DDGS import — support both duckduckgo-search 6.x and ddgs 7.x+
# ------------------------------------------------------------------

def _get_ddgs_class():
    try:
        from duckduckgo_search import DDGS
        return DDGS
    except ImportError:
        pass
    try:
        from ddgs import DDGS
        return DDGS
    except ImportError:
        return None


# ------------------------------------------------------------------
# Web search — SearXNG (primary) with DuckDuckGo fallback
# ------------------------------------------------------------------

_MAX_RETRIES = 3
_RETRY_BACKOFF = [1.0, 2.0, 4.0]


def _format_results(results: list[dict], source: str) -> str:
    """Format a list of {title, url, snippet} dicts into a labelled string."""
    lines = [f"[Search via {source}]"]
    for r in results:
        lines.append(
            f"Title: {r['title']}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r['snippet']}"
        )
    return "\n---\n".join(lines)


def _searxng_search(query: str, max_results: int) -> list[dict] | None:
    """
    Query the local SearXNG instance.
    Returns a list of result dicts, or None if SearXNG is unavailable.
    """
    import requests
    from src.config import SEARXNG_BASE_URL

    if not SEARXNG_BASE_URL:
        return None

    try:
        resp = requests.get(
            f"{SEARXNG_BASE_URL}/search",
            params={"q": query, "format": "json", "language": "en"},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.debug("SearXNG unavailable: %s", e)
        return None

    results = []
    for r in data.get("results", [])[:max_results]:
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
        })
    return results if results else None


def _brave_search(query: str, max_results: int) -> list[dict] | None:
    """
    Query Brave Search API (free tier ~1000 req/month).
    Returns a list of result dicts, or None if API key is not set or request fails.
    Get a free key at: https://api-dashboard.search.brave.com
    """
    import requests
    from src.config import BRAVE_SEARCH_API_KEY

    if not BRAVE_SEARCH_API_KEY:
        return None

    try:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": max_results, "search_lang": "en"},
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": BRAVE_SEARCH_API_KEY,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Brave Search failed: %s", e)
        return None

    results = []
    for r in data.get("web", {}).get("results", [])[:max_results]:
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("description", ""),
        })
    return results if results else None


def _ddg_search(query: str, max_results: int) -> list[dict] | None:
    """
    Query DuckDuckGo with retries.
    Returns a list of result dicts, or None if unavailable/failed.
    """
    DDGS = _get_ddgs_class()
    if DDGS is None:
        return None

    last_error: Exception | None = None
    for attempt, backoff in enumerate((_RETRY_BACKOFF + [0])[:_MAX_RETRIES]):
        try:
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=max_results))
            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
                for r in raw
            ]
            return results if results else None
        except Exception as e:
            last_error = e
            logger.warning("DDG search attempt %d failed: %s", attempt + 1, e)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(backoff)

    logger.error("DDG search failed after %d attempts: %s", _MAX_RETRIES, last_error)
    return None


def _web_search(query: str) -> str:
    from src.config import MAX_SEARCH_RESULTS
    from datetime import date

    today = date.today().strftime("%B %d %Y")
    dated_query = f"{query} {today}"

    # 1. SearXNG — fully local, no rate limits (requires Docker or native install)
    results = _searxng_search(dated_query, MAX_SEARCH_RESULTS)
    if results is not None:
        return _format_results(results, "SearXNG")

    # 2. Brave Search — reliable API, free tier ~1000 req/month (set BRAVE_SEARCH_API_KEY)
    results = _brave_search(dated_query, MAX_SEARCH_RESULTS)
    if results is not None:
        return _format_results(results, "Brave Search")

    # 3. DuckDuckGo — free but can be rate-limited
    results = _ddg_search(dated_query, MAX_SEARCH_RESULTS)
    if results is not None:
        return _format_results(results, "DuckDuckGo")

    return (
        "Web search unavailable. Options to fix:\n"
        "  1. SearXNG: docker compose up -d  (or run natively)\n"
        "  2. Brave Search: set BRAVE_SEARCH_API_KEY env var (free at api-dashboard.search.brave.com)\n"
        "  3. DuckDuckGo: pip install ddgs  (may be rate-limited)"
    )


web_search_tool = Tool(
    name="web_search",
    func=_web_search,
    description=(
        "Search the web. Auto-selects the best available backend: "
        "SearXNG (local, no rate limits) → Brave Search API → DuckDuckGo. "
        "Today's date is automatically appended so results are current. "
        "Input: a search query string. "
        "Output: titles, URLs, and snippets of the top results."
    ),
)

# ------------------------------------------------------------------
# Fetch URL  (with smart extraction + chunked summarisation)
# ------------------------------------------------------------------

_CHUNK_SIZE = 6_000
_MAX_SCRAPE = 60_000


def _extract_text(html: str) -> str:
    from bs4 import BeautifulSoup
    import re

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "button", "noscript", "iframe"]):
        tag.decompose()

    main = (
        soup.find("main") or
        soup.find("article") or
        soup.find(id="content") or
        soup.find(id="main") or
        soup.find(class_="content") or
        soup.find(class_="main") or
        soup.body or
        soup
    )

    lines = []
    for el in main.descendants:
        if el.name in ("h1", "h2", "h3", "h4"):
            t = el.get_text(strip=True)
            if t:
                lines.append(f"\n## {t}\n")
        elif el.name in ("p", "li", "td", "th", "pre", "code", "dd", "dt"):
            t = el.get_text(strip=True)
            if t:
                lines.append(t)

    text = "\n".join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


def _fetch_url(url: str) -> str:
    import requests

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        resp = requests.get(url, timeout=15, headers=headers, verify=False)
        resp.raise_for_status()
    except Exception as e:
        return f"Fetch error: {e}"

    content_type = resp.headers.get("Content-Type", "")
    if "json" in content_type:
        return resp.text[:_MAX_SCRAPE]
    if "text/plain" in content_type:
        return resp.text[:_MAX_SCRAPE]

    text = _extract_text(resp.text)
    if not text.strip():
        return "Page fetched but no readable text found (may require JavaScript or login)."

    total = len(text)
    if total <= _CHUNK_SIZE:
        return f"[Scraped {total} chars from {url}]\n\n{text}"

    chunks = [text[i:i+_CHUNK_SIZE] for i in range(0, min(total, _MAX_SCRAPE), _CHUNK_SIZE)]
    header = f"[Scraped {total} chars from {url} — {len(chunks)} chunks]\n\n"
    return header + "\n\n---CHUNK BREAK---\n\n".join(chunks)


fetch_url_tool = Tool(
    name="fetch_url",
    func=_fetch_url,
    description=(
        "Scrape and extract text from any URL — works for public websites AND "
        "internal network addresses (e.g. http://10.52.1.9). "
        "Automatically strips menus/ads and focuses on main content. "
        "Input: a URL or bare IP address (scheme optional). "
        "Output: clean extracted text ready for summarisation."
    ),
)

# ------------------------------------------------------------------
# File tools
# ------------------------------------------------------------------

def _read_file(path: str) -> str:
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"File not found: {path}"
        if p.stat().st_size > 1_000_000:
            return f"File too large to read (>{1_000_000} bytes): {path}"
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Read error: {e}"


read_file_tool = Tool(
    name="read_file",
    func=_read_file,
    description=(
        "Read the contents of a local file. "
        "Input: absolute or relative file path. "
        "Output: file text content."
    ),
)


def _list_dir(path: str = ".") -> str:
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"Path not found: {path}"
        entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
        lines = []
        for e in entries:
            kind = "FILE" if e.is_file() else "DIR "
            size = f"{e.stat().st_size:>10} bytes" if e.is_file() else ""
            lines.append(f"[{kind}] {e.name}  {size}")
        return "\n".join(lines) if lines else "Empty directory."
    except Exception as e:
        return f"List error: {e}"


list_dir_tool = Tool(
    name="list_dir",
    func=_list_dir,
    description=(
        "List files and subdirectories in a local directory. "
        "Input: directory path (default '.'). "
        "Output: directory listing."
    ),
)

# ------------------------------------------------------------------
# Memory tools
# ------------------------------------------------------------------

_memory_store_instance = None


def _get_memory():
    global _memory_store_instance
    if _memory_store_instance is None:
        from src.memory import MemoryStore
        _memory_store_instance = MemoryStore()
    return _memory_store_instance


def _memory_save(text: str) -> str:
    mem = _get_memory()
    result = mem.add(text, source="agent")
    return result  # already returns a message string


def _memory_recall(query: str) -> str:
    mem = _get_memory()
    return mem.search_text(query, k=5)


memory_store_tool = Tool(
    name="memory_store",
    func=_memory_save,
    description=(
        "Save important information to persistent vector memory. "
        "Input: the text or fact to remember. "
        "Output: confirmation (or note if a similar memory already exists)."
    ),
)

memory_recall_tool = Tool(
    name="memory_recall",
    func=_memory_recall,
    description=(
        "Search persistent memory for relevant past information. "
        "Input: a query describing what you want to recall. "
        "Output: the most relevant stored memories (only those above relevance threshold)."
    ),
)

# ------------------------------------------------------------------
# SFIS query (internal manufacturing system at 10.52.1.9)
# ------------------------------------------------------------------

def _sfis_query(inp: str) -> str:
    """
    Input format:  "SN123456"
               or  "SN123456, R2251"   (SN + component location for vendor data)
    """
    from src.sfis import query_sn, SFISAuthError
    parts = [p.strip() for p in inp.split(",", 1)]
    sn = parts[0]
    component = parts[1] if len(parts) > 1 else None

    if not sn:
        print("[SFIS] ERROR — sfis_query called with empty SN, rejecting")
        return (
            "Error: serial number is empty. "
            "Provide the SN as Action Input, e.g. 'Action Input: HMHHTX00E960000LQ7'. "
            "Do not call sfis_query without a serial number."
        )
    try:
        result = query_sn(sn, component)
    except SFISAuthError as e:
        return f"SFIS auth error: {e}"
    except Exception as e:
        return f"SFIS query error: {e}"

    # Auto-cache full SFIS data to memory so follow-up questions don't need a re-query.
    # The 0.90 near-duplicate threshold in MemoryStore prevents duplicate entries for
    # the same SN queried twice.
    if "SFIS Data for SN" in result:
        try:
            from src.memory import MemoryStore
            MemoryStore().add(result, source="sfis_cache")
            print(f"[SFIS] Cached SN={sn} data to memory ({len(result)} chars)")
        except Exception as e:
            print(f"[SFIS] Warning: could not save to memory: {e}")

    return result


sfis_query_tool = Tool(
    name="sfis_query",
    func=_sfis_query,
    description=(
        "Query the internal SFIS manufacturing system (http://10.52.1.9) for a serial number. "
        "Automatically checks connectivity, authenticates, and validates the SN — "
        "returns a clear message if the server is unreachable, login fails, or the SN is not found. "
        "Returns structured fields: Phase, Model, Config, SMT Line, Panel SN, "
        "SN position in panel, Failed Date, Lab In Time, Group Name, "
        "Failure Message, and List of Failing Tests. "
        "Input format: 'SERIAL_NUMBER'  or  'SERIAL_NUMBER, COMPONENT_LOCATION' "
        "(add a component location like 'R2251' or 'U7000' to also get vendor/lot/date-code data). "
        "Credentials must be saved in sfis_cred.json in the project root."
    ),
)


# ------------------------------------------------------------------
# SFIS 2A defect query (time period)
# ------------------------------------------------------------------

def _sfis_2a_query(inp: str) -> str:
    """
    Input: comma-separated key=value pairs.
    Required: from_date, to_date
    Optional: model_name, model_serial, line_name, group_name, error_code, mo, retest_sequence
    """
    from src.sfis import query_2a_defects, SFISAuthError

    params: dict[str, str] = {}
    for part in inp.split(","):
        part = part.strip()
        if "=" in part:
            k, _, v = part.partition("=")
            params[k.strip()] = v.strip()

    from_date = params.get("from_date", "")
    to_date = params.get("to_date", "")
    if not from_date or not to_date:
        return (
            "Error: both from_date and to_date are required. "
            "Example: 'from_date=2026/05/12, to_date=2026/05/12'"
        )

    try:
        return query_2a_defects(
            from_date=from_date,
            to_date=to_date,
            model_name=params.get("model_name", ""),
            model_serial=params.get("model_serial", ""),
            line_name=params.get("line_name", ""),
            group_name=params.get("group_name", "ALL"),
            error_code=params.get("error_code", ""),
            mo=params.get("mo", "ALL"),
            retest_sequence=params.get("retest_sequence", "FIRST"),
        )
    except SFISAuthError as e:
        return f"SFIS auth error: {e}"
    except Exception as e:
        return f"SFIS 2A query error: {e}"


sfis_2a_tool = Tool(
    name="sfis_2a_defects",
    func=_sfis_2a_query,
    description=(
        "Query SFIS 2A defect data for a date/time range. "
        "Returns group name, test time, error codes, and defect records. "
        "Input: comma-separated key=value pairs. "
        "Required: from_date, to_date — MUST include HH:MM e.g. '2026/06/03 00:00' and '2026/06/03 23:59'. "
        "For large result sets (>200 records) the full data is automatically saved to an Excel file "
        "in the output/ folder and a statistical summary (breakdown by group, type, top error codes) "
        "is returned so the LLM can answer questions without being overwhelmed by raw data. "
        "Optional: model_name, model_serial, line_name, group_name (default ALL), "
        "error_code, mo (default ALL), retest_sequence (default FIRST). "
        "Example: 'from_date=2026/05/12 00:00, to_date=2026/05/12 23:59, model_name=XY1234'"
    ),
)


# ------------------------------------------------------------------
# SFIS PVS-vs-SFIS query (component / vendor traceability)
# ------------------------------------------------------------------

def _sfis_pvs_query(inp: str) -> str:
    """
    Input: comma-separated key=value pairs.
    Optional keys: sn, location, model_name, family, from_date, to_date, mo, carton_no, comp_pn
    """
    from src.sfis import query_pvs, SFISAuthError

    params: dict[str, str] = {}
    for part in inp.split(","):
        part = part.strip()
        if "=" in part:
            k, _, v = part.partition("=")
            params[k.strip()] = v.strip()

    try:
        return query_pvs(
            sn=params.get("sn", ""),
            location=params.get("location", ""),
            model_name=params.get("model_name", ""),
            family=params.get("family", ""),
            from_date=params.get("from_date", ""),
            to_date=params.get("to_date", ""),
            mo=params.get("mo", ""),
            carton_no=params.get("carton_no", ""),
            comp_pn=params.get("comp_pn", ""),
        )
    except SFISAuthError as e:
        return f"SFIS auth error: {e}"
    except Exception as e:
        return f"SFIS PVS query error: {e}"


sfis_pvs_tool = Tool(
    name="sfis_pvs_query",
    func=_sfis_pvs_query,
    description=(
        "Query SFIS PVS-vs-SFIS for component vendor traceability data. "
        "Returns all fields for each matched record: SERIAL_NUMBER, GROUP_NAME, MO_NUMBER, "
        "REEL_ID, SEAT, COMP_PART_NO, PROJECT_VERSION, VENDOR, LOT_NO, DATE_CODE, LOCATION, etc. "
        "MUST provide at least: sn (serial number) AND location (component reference designator). "
        "When sn is provided, disable_period is applied automatically — no date filter needed. "
        "Input: comma-separated key=value pairs. "
        "Keys: sn, location (e.g. U7000), model_name, family, from_date, to_date, mo, carton_no, comp_pn. "
        "Example: 'sn=HMHHTX00E960000LQ7, location=U7000'"
    ),
)


# ------------------------------------------------------------------
# Exported tool list
# ------------------------------------------------------------------

ALL_TOOLS = [
    web_search_tool,
    fetch_url_tool,
    read_file_tool,
    list_dir_tool,
    memory_store_tool,
    memory_recall_tool,
    sfis_query_tool,
    sfis_2a_tool,
    sfis_pvs_tool,
]
