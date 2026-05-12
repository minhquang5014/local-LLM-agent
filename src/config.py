import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model" / "Qwen3.5B-9B"

# HuggingFace transformers settings
HF_MODEL_PATH = str(MODEL_DIR)
MAX_NEW_TOKENS = 512
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
MAX_SEARCH_RESULTS = 5

# Agent
MAX_ITERATIONS = 10
AGENT_VERBOSE = True
