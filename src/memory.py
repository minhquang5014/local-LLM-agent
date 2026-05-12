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
import uuid
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings

from src.config import CHROMA_DIR, COLLECTION_NAME

logger = logging.getLogger(__name__)


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
        doc_id = str(uuid.uuid4())
        meta = {"source": source}
        if metadata:
            meta.update(metadata)
        self._collection.add(
            ids=[doc_id],
            documents=[text],
            metadatas=[meta],
        )
        logger.debug("Memory added (id=%s, source=%s)", doc_id, source)
        return doc_id

    def add_many(self, texts: List[str], source: str = "agent") -> List[str]:
        ids = [str(uuid.uuid4()) for _ in texts]
        metas = [{"source": source}] * len(texts)
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
    ) -> List[Dict[str, Any]]:
        """Return the top-k most relevant memory entries."""
        kwargs: Dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(k, max(1, self._collection.count())),
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
            hits.append({"text": doc, "metadata": meta, "score": 1 - dist})
        return hits

    def search_text(self, query: str, k: int = 5) -> str:
        """Convenience: return a formatted string of top-k memories."""
        hits = self.search(query, k=k)
        if not hits:
            return "No relevant memories found."
        lines = []
        for i, h in enumerate(hits, 1):
            lines.append(f"[{i}] (score={h['score']:.3f}) {h['text']}")
        return "\n".join(lines)

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
