"""
memory_os/store.py  —  Week 1: MemoryStore (Qdrant backend)
============================================================
Every memory carries a full "memory card" payload:
  - text            : original raw text
  - importance      : float 0-1
  - emotional_score : float 0-1  (filled by EmotionTagger)
  - emotional_label : str        (joy/fear/anger/sadness/neutral/surprise)
  - created_at      : unix timestamp
  - last_accessed   : unix timestamp
  - access_count    : int
  - access_history  : list[float]  timestamps of every retrieval (for Ebbinghaus)
  - memory_type     : "episodic" | "semantic" | "working"
  - source          : who/what created this memory (provenance)
"""

import os
import time
import uuid
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    PointIdsList,
)

load_dotenv()

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "memories")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL",   "text-embedding-3-small")
VECTOR_DIM       = int(os.getenv("VECTOR_DIM",     "1536"))


def _embed(text: str, client: OpenAI) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


class MemoryStore:
    """
    Core storage layer. Connects to Qdrant and manages episodic memories.
    Every memory gets a full psychological metadata card.

    Usage:
        store = MemoryStore()
        mid = store.insert("User prefers concise answers.", importance=0.8)
        results = store.retrieve("What does user prefer?", top_k=50)
    """

    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        qdrant_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        collection: str = COLLECTION_NAME,
    ):
        self.collection = collection
        self._oai = OpenAI(api_key=openai_key or os.getenv("OPENAI_API_KEY"))
        self._qdrant = QdrantClient(
            url=qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=qdrant_key or os.getenv("QDRANT_API_KEY"),
        )
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self._qdrant.get_collections().collections]
        if self.collection not in existing:
            self._qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )

    def insert(
        self,
        text: str,
        importance: float = 0.5,
        memory_type: str = "episodic",
        emotional_score: float = 0.0,
        emotional_label: str = "neutral",
        source: str = "user",
        extra_payload: Optional[dict] = None,
    ) -> str:
        """Embed text and store with full memory card. Returns UUID."""
        vector = _embed(text, self._oai)
        memory_id = str(uuid.uuid4())
        now = time.time()

        payload = {
            "text":            text,
            "importance":      float(importance),
            "emotional_score": float(emotional_score),
            "emotional_label": emotional_label,
            "created_at":      now,
            "last_accessed":   now,
            "access_count":    0,
            "access_history":  [],   # list of timestamps — used by Ebbinghaus
            "memory_type":     memory_type,
            "source":          source,
        }
        if extra_payload:
            payload.update(extra_payload)

        self._qdrant.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=memory_id, vector=vector, payload=payload)],
        )
        return memory_id

    def retrieve(
        self,
        query: str,
        top_k: int = 50,
        memory_type: Optional[str] = None,
        update_access: bool = True,
    ):
        """
        Retrieve top_k most similar memories.
        Returns raw cosine similarity order — DecayReranker re-ranks these.
        """
        query_vector = _embed(query, self._oai)

        search_filter = None
        if memory_type:
            search_filter = Filter(
                must=[FieldCondition(key="memory_type", match=MatchValue(value=memory_type))]
            )

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

    def get_by_id(self, memory_id: str):
        results = self._qdrant.retrieve(
            collection_name=self.collection,
            ids=[memory_id],
            with_payload=True,
            with_vectors=True,
        )
        return results[0] if results else None

    def update_payload(self, memory_id: str, updates: dict) -> None:
        self._qdrant.set_payload(
            collection_name=self.collection,
            payload=updates,
            points=[memory_id],
        )

    def delete(self, memory_id: str) -> None:
        self._qdrant.delete(
            collection_name=self.collection,
            points_selector=PointIdsList(points=[memory_id]),
        )

    def get_all(self, memory_type: Optional[str] = None, limit: int = 1000):
        scroll_filter = None
        if memory_type:
            scroll_filter = Filter(
                must=[FieldCondition(key="memory_type", match=MatchValue(value=memory_type))]
            )
        results, _ = self._qdrant.scroll(
            collection_name=self.collection,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )
        return results

    def count(self) -> int:
        return self._qdrant.count(collection_name=self.collection).count

    def wipe(self) -> None:
        self._qdrant.delete_collection(self.collection)
        self._ensure_collection()
