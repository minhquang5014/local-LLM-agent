"""
LangChain-compatible tools for the agent.

Available tools:
  - web_search      — DuckDuckGo search returning summaries
  - fetch_url       — Download and clean a web page
  - read_file       — Read a local file
  - list_dir        — List files in a directory
  - memory_store    — Save a fact to vector memory
  - memory_recall   — Retrieve relevant memories
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.tools import Tool

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Web search
# ------------------------------------------------------------------

def _web_search(query: str) -> str:
    from ddgs import DDGS
    from src.config import MAX_SEARCH_RESULTS
    from datetime import date
    today = date.today().strftime("%B %d %Y")
    dated_query = f"{query} {today}"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(dated_query, max_results=MAX_SEARCH_RESULTS))
        if not results:
            return "No results found."
        lines = []
        for r in results:
            lines.append(f"Title: {r.get('title', '')}\nURL: {r.get('href', '')}\nSnippet: {r.get('body', '')}\n")
        return "\n---\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"


web_search_tool = Tool(
    name="web_search",
    func=_web_search,
    description=(
        "Search the web using DuckDuckGo. Today's date is automatically appended "
        "to every query so results are always current. "
        "Input: a search query string. "
        "Output: titles, URLs, and snippets of the top results."
    ),
)

# ------------------------------------------------------------------
# Fetch URL  (with smart extraction + chunked summarisation)
# ------------------------------------------------------------------

# Max chars fed to the LLM per chunk. Fits comfortably in Qwen3.5 context.
_CHUNK_SIZE = 6_000
# Hard cap on total chars scraped (prevents runaway pages)
_MAX_SCRAPE = 60_000


def _extract_text(html: str) -> str:
    """Return clean text from HTML, preferring main-content tags."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    # Remove noise tags
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "button", "noscript", "iframe"]):
        tag.decompose()

    # Try to isolate the main content block
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
    # Collapse excessive blank lines
    import re
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


def _fetch_url(url: str) -> str:
    import requests

    # Support bare IPs / internal addresses (add http:// if missing scheme)
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/124.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        resp = requests.get(url, timeout=15, headers=headers, verify=False)
        resp.raise_for_status()
    except Exception as e:
        return f"Fetch error: {e}"

    content_type = resp.headers.get("Content-Type", "")

    # Plain text / JSON — return directly
    if "json" in content_type:
        return resp.text[:_MAX_SCRAPE]
    if "text/plain" in content_type:
        return resp.text[:_MAX_SCRAPE]

    text = _extract_text(resp.text)

    if not text.strip():
        return "Page fetched but no readable text found (may require JavaScript or login)."

    total = len(text)

    # Short page — return as-is
    if total <= _CHUNK_SIZE:
        return f"[Scraped {total} chars from {url}]\n\n{text}"

    # Long page — return first chunk + summary prompt hint
    chunks = [text[i:i+_CHUNK_SIZE] for i in range(0, min(total, _MAX_SCRAPE), _CHUNK_SIZE)]
    header = f"[Scraped {total} chars from {url} — split into {len(chunks)} chunks. Showing all chunks.]\n\n"
    return header + "\n\n---CHUNK BREAK---\n\n".join(chunks)


fetch_url_tool = Tool(
    name="fetch_url",
    func=_fetch_url,
    description=(
        "Scrape and extract text from any URL — works for public websites AND "
        "internal network addresses (e.g. http://10.52.1.9). "
        "Automatically strips menus/ads and focuses on main content. "
        "Handles large pages by returning all content in chunks. "
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
    doc_id = mem.add(text, source="agent")
    return f"Saved to memory (id={doc_id})."


def _memory_recall(query: str) -> str:
    mem = _get_memory()
    return mem.search_text(query, k=5)


memory_store_tool = Tool(
    name="memory_store",
    func=_memory_save,
    description=(
        "Save important information to persistent vector memory. "
        "Input: the text or fact to remember. "
        "Output: confirmation with memory ID."
    ),
)

memory_recall_tool = Tool(
    name="memory_recall",
    func=_memory_recall,
    description=(
        "Search persistent memory for relevant past information. "
        "Input: a query describing what you want to recall. "
        "Output: the most relevant stored memories."
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
    try:
        return query_sn(sn, component)
    except SFISAuthError as e:
        return f"SFIS auth error: {e}"
    except Exception as e:
        return f"SFIS query error: {e}"


sfis_query_tool = Tool(
    name="sfis_query",
    func=_sfis_query,
    description=(
        "Query the internal SFIS manufacturing system (http://10.52.1.9) for a serial number. "
        "Automatically checks server connectivity, authenticates, and validates the SN — "
        "returns a clear message if the server is unreachable, login fails, or the SN is not found. "
        "Returns traveler data: phase, model, config, SMT line, panel SN, failure history, "
        "group name, failure message, and failing tests. "
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
        "Required: from_date, to_date (format 'YYYY/MM/DD' or 'YYYY/MM/DD HH:MM'). "
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
        "Returns vendor, lot number, date code, component SN, and location for each match. "
        "Input: comma-separated key=value pairs (all optional but at least one filter is needed). "
        "Keys: sn (serial number), location (e.g. U7000), model_name, family, "
        "from_date, to_date, mo, carton_no, comp_pn. "
        "Example: 'sn=HMHHL400B0V0000LQ7, location=U7000'"
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
