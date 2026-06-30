import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model" / "Qwen3.5B-9B"

# HuggingFace transformers settings
HF_MODEL_PATH = str(MODEL_DIR)
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# ChromaDB
CHROMA_DIR = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "agent_memory"

# Ollama (if installed later)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# Web search
MAX_SEARCH_RESULTS = 6

# ── External Web Bridge (Mac #2) ──────────────────────────────────────
# The agent runs inside the 10.52 corporate network with NO internet access,
# so web_search and fetch_url (for PUBLIC URLs) are delegated to the web bridge
# on Mac #2 over the private Thunderbolt link. Mac #2 does the SearXNG search +
# scrape + trim and returns LLM-ready text.
#   - EXTERNAL_BRIDGE_URL: base URL of the bridge_api service on Mac #2
#   - EXTERNAL_BRIDGE_API_KEY: shared secret sent as the X-API-Key header
# Override both via environment / .env in other deployments.
EXTERNAL_BRIDGE_URL = os.getenv("EXTERNAL_BRIDGE_URL", "http://192.168.100.2:8000")
EXTERNAL_BRIDGE_API_KEY = os.getenv("EXTERNAL_BRIDGE_API_KEY", "fa-lab-ai-data-center")
# How long to wait for the bridge (search+scrape of several pages can take a while).
EXTERNAL_BRIDGE_TIMEOUT = float(os.getenv("EXTERNAL_BRIDGE_TIMEOUT", "90"))

# Legacy local search backends (used only if you run the agent on a machine WITH
# internet and no bridge — kept for dev/laptop testing). Order: SearXNG → Brave → DDG.
SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "http://localhost:8080")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY", "")

# Agent
MAX_ITERATIONS = 10
AGENT_VERBOSE = True
