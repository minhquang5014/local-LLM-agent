"""
Persistent vector memory backed by ChromaDB.

Stores text snippets with metadata so the agent can retrieve
relevant context from previous runs.

Usage:
    from src.memory import MemoryStore
    mem = MemoryStore()
    mem.add("Paris is the capital of France.", source="user")
    results = mem.search("capital of France")
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings

from src.config import CHROMA_DIR, COLLECTION_NAME

logger = logging.getLogger(__name__)

# Memories below this cosine-similarity score are considered irrelevant
_RELEVANCE_THRESHOLD = 0.30

# Two memories with similarity above this are considered near-duplicates
_DEDUP_THRESHOLD = 0.90


class MemoryStore:
    def __init__(self, collection_name: str = COLLECTION_NAME):
        self._client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "MemoryStore ready — collection '%s' (%d docs)",
            collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        source: str = "agent",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save *text* to memory.  Returns a status string.
        Skips saving if a near-duplicate already exists.
        """
        text = text.strip()
        if not text:
            return "Empty text — nothing saved."

        # Near-duplicate check (only if we have existing docs)
        if self._collection.count() > 0:
            try:
                hits = self.search(text, k=1)
                if hits and hits[0]["score"] >= _DEDUP_THRESHOLD:
                    return (
                        f"Near-duplicate found (score={hits[0]['score']:.3f}) — not saved. "
                        f"Existing: {hits[0]['text'][:120]}"
                    )
            except Exception:
                pass  # if dedup check fails, continue saving

        doc_id = str(uuid.uuid4())
        meta: Dict[str, Any] = {
            "source": source,
            "timestamp": int(time.time()),
        }
        if metadata:
            meta.update(metadata)

        self._collection.add(
            ids=[doc_id],
            documents=[text],
            metadatas=[meta],
        )
        logger.debug("Memory added (id=%s, source=%s)", doc_id, source)
        return f"Saved to memory (id={doc_id})."

    def add_many(self, texts: List[str], source: str = "agent") -> List[str]:
        now = int(time.time())
        ids = [str(uuid.uuid4()) for _ in texts]
        metas = [{"source": source, "timestamp": now}] * len(texts)
        self._collection.add(ids=ids, documents=texts, metadatas=metas)
        return ids

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        threshold: float = _RELEVANCE_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        """
        Return the top-k most relevant memory entries above *threshold*.
        Results are sorted by score descending.
        """
        count = self._collection.count()
        if count == 0:
            return []

        kwargs: Dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(k, count),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1.0 - dist  # cosine distance → similarity
            if score >= threshold:
                hits.append({"text": doc, "metadata": meta, "score": score})

        # Sort by score descending (ChromaDB already returns sorted, but be explicit)
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits

    def search_text(self, query: str, k: int = 5) -> str:
        """Convenience: return a formatted string of relevant memories."""
        hits = self.search(query, k=k)
        if not hits:
            return "No relevant memories found."
        lines = []
        for i, h in enumerate(hits, 1):
            ts = h["metadata"].get("timestamp")
            age = ""
            if ts:
                elapsed = int(time.time()) - ts
                if elapsed < 3600:
                    age = f" | {elapsed // 60}m ago"
                elif elapsed < 86400:
                    age = f" | {elapsed // 3600}h ago"
                else:
                    age = f" | {elapsed // 86400}d ago"
            lines.append(f"[{i}] (relevance={h['score']:.2f}{age}) {h['text']}")
        return "\n".join(lines)

    def recent(self, k: int = 10) -> List[Dict[str, Any]]:
        """Return the *k* most recently added memories."""
        if self._collection.count() == 0:
            return []
        raw = self._collection.get(include=["documents", "metadatas"])
        items = [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(raw["documents"], raw["metadatas"])
        ]
        items.sort(key=lambda x: x["metadata"].get("timestamp", 0), reverse=True)
        return items[:k]

    # ------------------------------------------------------------------
    # Delete / reset
    # ------------------------------------------------------------------

    def delete(self, doc_id: str) -> None:
        self._collection.delete(ids=[doc_id])

    def clear(self) -> None:
        count = self._collection.count()
        if count:
            ids = self._collection.get()["ids"]
            self._collection.delete(ids=ids)
        logger.info("Memory cleared (%d docs removed)", count)

    def count(self) -> int:
        return self._collection.count()
