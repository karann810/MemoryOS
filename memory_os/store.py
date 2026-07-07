"""
memory_os/store.py  —  MemoryStore (Qdrant backend)
====================================================
Every memory carries a full "memory card" payload:
  - text            : original raw text
  - importance      : float 0-1
  - emotional_score : float 0-1
  - emotional_label : str  (joy/fear/anger/sadness/neutral/surprise)
  - created_at      : unix timestamp
  - last_accessed   : unix timestamp
  - access_count    : int
  - access_history  : list[float]  timestamps of every retrieval (Ebbinghaus)
  - memory_type     : "episodic" | "semantic" | "working"
  - source          : provenance string

Uses litellm for embeddings — works with OpenAI, Cohere, Voyage, etc.
"""

import os
import time
import uuid
from typing import Optional

try:
    import litellm
except Exception:
    from . import _litellm as litellm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, PointStruct, VectorParams,
    Filter, FieldCondition, MatchValue, PointIdsList,
)

# Dim lookup for common embedding models (auto-detected if not specified)
KNOWN_DIMS = {
    "text-embedding-3-small":        1536,
    "text-embedding-3-large":        3072,
    "text-embedding-ada-002":        1536,
    "openai/text-embedding-3-small": 1536,
    "openai/text-embedding-3-large": 3072,
    "voyage-large-2":                1536,
    "cohere/embed-english-v3.0":     1024,
}

COLLECTION_NAME = "memories"


def _embed(text: str, model: str, api_key: str) -> list[float]:
    """Embed text via litellm — supports any embedding provider."""
    response = litellm.embedding(model=model, input=[text])
    return response.data[0]["embedding"]


def _get_vector_dim(model: str, api_key: str) -> int:
    """Detect embedding dimension by running a test embed."""
    if model in KNOWN_DIMS:
        return KNOWN_DIMS[model]
    # Unknown model — probe it
    vec = _embed("test", model, api_key)
    return len(vec)


class MemoryStore:
    """
    Core Qdrant storage layer. Works with any litellm-compatible embedding model.

    Usage:
        store = MemoryStore(
            qdrant_url    = "http://localhost:6333",
            embed_model   = "text-embedding-3-small",
            embed_api_key = "sk-...",
        )
        mid = store.insert("User prefers concise answers.", importance=0.8)
        results = store.retrieve("What does user prefer?", top_k=50)
    """

    def __init__(
        self,
        qdrant_url:    Optional[str] = None,
        embed_model:   str = "text-embedding-3-small",
        embed_api_key: str = "",
        qdrant_key:    Optional[str] = None,
        collection:    str = COLLECTION_NAME,
    ):
        self.embed_model   = embed_model
        self.embed_api_key = embed_api_key
        self.collection    = collection

        # If no Qdrant URL is provided, fall back to a simple in-memory store
        self._in_memory = False
        if not qdrant_url:
            self._in_memory = True
            self._mems = []
        else:
            self._qdrant = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_key,
            )
            self._vector_dim = _get_vector_dim(embed_model, embed_api_key)
            self._ensure_collection()

    def embed(self, text: str) -> list[float]:
        return _embed(text, self.embed_model, self.embed_api_key)

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self._qdrant.get_collections().collections]
        if self.collection not in existing:
            self._qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self._vector_dim, distance=Distance.COSINE),
            )

    def insert(
        self,
        text:            str,
        importance:      float = 0.5,
        memory_type:     str = "episodic",
        emotional_score: float = 0.0,
        emotional_label: str = "neutral",
        source:          str = "user",
        extra_payload:   Optional[dict] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Embed text and store with full memory card. Returns UUID."""
        vector    = self.embed(text)
        memory_id = str(uuid.uuid4())
        now       = time.time()

        payload = {
            "user_id":         user_id,
            "text":            text,
            "importance":      float(importance),
            "emotional_score": float(emotional_score),
            "emotional_label": emotional_label,
            "created_at":      now,
            "last_accessed":   now,
            "access_count":    0,
            "access_history":  [],
            "memory_type":     memory_type,
            "source":          source,
        }
        if extra_payload:
            payload.update(extra_payload)

        if self._in_memory:
            item = {"id": memory_id, "vector": None, "payload": payload}
            self._mems.append(item)
        else:
            self._qdrant.upsert(
                collection_name=self.collection,
                points=[PointStruct(id=memory_id, vector=vector, payload=payload)],
            )
        return memory_id

    def retrieve(
        self,
        query:         str,
        top_k:         int = 50,
        memory_type:   Optional[str] = None,
        user_id:       Optional[str] = None,
        update_access: bool = True,
    ):
        query_vector  = self.embed(query)
        search_filter = None
        must_conditions = []
        if memory_type:
            must_conditions.append(FieldCondition(key="memory_type", match=MatchValue(value=memory_type)))
        if user_id is not None:
            must_conditions.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
        if must_conditions:
            search_filter = Filter(must=must_conditions)

        results = self._qdrant.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=search_filter,
            with_payload=True,
        )

        if update_access:
            now = time.time()
            for r in results:
                history = r.payload.get("access_history", [])
                history.append(now)
                self._qdrant.set_payload(
                    collection_name=self.collection,
                    payload={
                        "access_count":   r.payload.get("access_count", 0) + 1,
                        "last_accessed":  now,
                        "access_history": history,
                    },
                    points=[r.id],
                )
        return results

        
    def _fake_hit(self, item):
        class H:
            pass
        h = H()
        h.id = item["id"]
        h.payload = item["payload"]
        return h

    def retrieve(self,
        query:         str,
        top_k:         int = 50,
        memory_type:   Optional[str] = None,
        user_id:       Optional[str] = None,
        update_access: bool = True,
    ):
        if not self._in_memory:
            return self._qdrant.search(
                collection_name=self.collection,
                query_vector=self.embed(query),
                limit=top_k,
                query_filter=None,
                with_payload=True,
            )
        # Simple substring search over stored payload text
        matches = []
        q = query.lower()
        for item in self._mems:
            if memory_type and item["payload"].get("memory_type") != memory_type:
                continue
            if user_id is not None and item["payload"].get("user_id") != user_id:
                continue
            if q in item["payload"].get("text", "").lower():
                # update access history if requested
                if update_access:
                    now = time.time()
                    history = item["payload"].get("access_history", [])
                    history.append(now)
                    item["payload"]["access_history"] = history
                    item["payload"]["access_count"] = item["payload"].get("access_count", 0) + 1
                    item["payload"]["last_accessed"] = now
                matches.append(self._fake_hit(item))
        return matches

    def get_by_id(self, memory_id: str):
        if not self._in_memory:
            results = self._qdrant.retrieve(
                collection_name=self.collection,
                ids=[memory_id],
                with_payload=True,
                with_vectors=True,
            )
            return results[0] if results else None
        for item in self._mems:
            if item["id"] == memory_id:
                return item
        return None

    def update_payload(self, memory_id: str, updates: dict) -> None:
        if not self._in_memory:
            self._qdrant.set_payload(
                collection_name=self.collection,
                payload=updates,
                points=[memory_id],
            )
        else:
            for item in self._mems:
                if item["id"] == memory_id:
                    item["payload"].update(updates)
                    return

    def delete(self, memory_id: str) -> None:
        if not self._in_memory:
            self._qdrant.delete(
                collection_name=self.collection,
                points_selector=PointIdsList(points=[memory_id]),
            )
        else:
            self._mems = [m for m in self._mems if m["id"] != memory_id]

    def get_all(self, memory_type: Optional[str] = None, user_id: Optional[str] = None, limit: int = 1000):
        scroll_filter = None
        must_conditions = []
        if memory_type:
            must_conditions.append(FieldCondition(key="memory_type", match=MatchValue(value=memory_type)))
        if user_id is not None:
            must_conditions.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
        if must_conditions:
            scroll_filter = Filter(must=must_conditions)
        if not self._in_memory:
            results, _ = self._qdrant.scroll(
                collection_name=self.collection,
                scroll_filter=scroll_filter,
                limit=limit,
                with_payload=True,
                with_vectors=True,
            )
            return results
        return [self._fake_hit(m) for m in self._mems]

    def count(self) -> int:
        if not self._in_memory:
            return self._qdrant.count(collection_name=self.collection).count
        return len(self._mems)

    def wipe(self) -> None:
        if not self._in_memory:
            self._qdrant.delete_collection(self.collection)
            self._ensure_collection()
        else:
            self._mems = []

    def upsert_vector(self, vector: list[float], payload: dict) -> str:
        """Upsert a raw vector with payload. Returns generated id."""
        memory_id = str(uuid.uuid4())
        now = time.time()
        stored_payload = payload.copy() if payload else {}
        stored_payload.setdefault("created_at", now)
        stored_payload.setdefault("last_accessed", now)
        self._qdrant.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=memory_id, vector=vector, payload=stored_payload)],
        )
        return memory_id

    def store_interaction(
        self,
        user_id: Optional[str],
        query: str,
        response: str,
        importance: float = 0.4,
        memory_type: str = "episodic",
        response_importance: float = 0.3,
        source: str = "interaction",
        extra_payload: Optional[dict] = None,
    ) -> list[str]:
        """Store a user query and LLM response as two memories. Returns list of ids.

        This creates two memory records: one for the user query (higher importance)
        and one for the assistant response (lower importance).
        """
        qid = self.insert(
            text=f"User asked: {query}",
            importance=importance,
            memory_type=memory_type,
            source=source,
            extra_payload=extra_payload,
            user_id=user_id,
        )
        rid = self.insert(
            text=f"Assistant replied: {response}",
            importance=response_importance,
            memory_type=memory_type,
            source=source,
            extra_payload=extra_payload,
            user_id=user_id,
        )
        return [qid, rid]
