"""Final public MemoryOS facade.

The host application owns answer generation. MemoryOS extracts important memory
chunks from the user's prompt, stores them separately in Qdrant, and keeps a
small rolling prompt/response history per session.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from typing import Any

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.exceptions import UnexpectedResponse
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointIdsList,
        PointStruct,
        VectorParams,
    )
except Exception:
    QdrantClient = None
    UnexpectedResponse = Exception
    Distance = None
    FieldCondition = None
    Filter = None
    MatchValue = None
    PointIdsList = None
    PointStruct = None
    VectorParams = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


COLLECTION_NAME = "memory_os"
SESSION_PAIR_CAP = 7
RETRIEVE_LIMIT = 12
RETURN_MEMORY_LIMIT = 5
RECENT_PAIR_LIMIT = 5
SECONDS_PER_DAY = 86400.0
DEFAULT_STABILITY = 2.0
STABILITY_GROWTH = 0.75
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

EMOTION_WEIGHTS = {
    "joy": 1.15,
    "sadness": 1.2,
    "anger": 1.25,
    "fear": 1.35,
    "surprise": 1.1,
    "love": 1.25,
    "neutral": 1.0,
}


EXTRACTION_PROMPT = """You extract important memories from a user's prompt.

Read only the user prompt and break it into separate memory items when multiple
important topics appear. Each item should be independently useful for future
retrieval and embedding.

For each memory item, also assign:
- importance: float from 0.0 to 1.0
- emotion: one of joy, sadness, anger, fear, surprise, love, neutral

Rules:
- Do not answer the prompt.
- Do not use the assistant response.
- Split different preferences, goals, constraints, projects, facts, or plans
  into separate memory items.
- Keep each item short and standalone.
- Ignore filler, greetings, and temporary wording that is not useful later.
- Return JSON only.

Output format:
{{"memories":[{{"text":"memory 1","importance":0.9,"emotion":"neutral"}}]}}

If nothing should be stored, return:
{{"memories":[]}}

User prompt:
{prompt}
"""


class MemoryOS:
    """Small public API: store prompt memories, retrieve context."""

    _session_pairs: dict[str, list[dict[str, Any]]] = {}

    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        llm: Any,
        session_id: str,
    ) -> None:
        missing = [
            name
            for name, value in {
                "qdrant_url": qdrant_url,
                "llm": llm,
                "session_id": session_id,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required MemoryOS config: {', '.join(missing)}")
        if not hasattr(llm, "invoke"):
            raise TypeError("llm must provide an invoke(prompt: str) method")
        if QdrantClient is None:
            raise ImportError(
                "MemoryOS requires qdrant-client. Install the package dependencies before use."
            )
        if SentenceTransformer is None:
            raise ImportError(
                "MemoryOS requires sentence-transformers. Install package dependencies before use."
            )

        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.llm = llm
        self.session_id = session_id
        self.collection = COLLECTION_NAME
        self._client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None)
        self._collection_ready = False
        self._payload_indexes_ready = False
        self._session_pairs.setdefault(session_id, [])
        self._embedder = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)

        # Eagerly create the collection + payload indexes now, instead of
        # waiting for the first successful .store() call. This guarantees
        # .retrieve() never hits a "missing index" 400 error, even if it's
        # called before any .store(), or in a fresh process/kernel.
        vector_size = self._embedder.get_sentence_embedding_dimension()
        self._ensure_collection(vector_size)

    def store(self, prompt: str, response: str) -> None:
        """Extract prompt memories into Qdrant and keep the raw pair in session history."""
        if not prompt or not response:
            return

        pair_id = str(uuid.uuid4())
        now = time.time()
        memories = self._extract_memories(prompt)

        if memories:
            vectors = [self._embed(memory["text"]) for memory in memories]
            self._ensure_collection(len(vectors[0]))
            points = []
            for memory, vector in zip(memories, vectors):
                emotion = self._normalize_emotion(memory.get("emotion"))
                importance = self._normalize_importance(memory.get("importance"))
                emotional_weight = EMOTION_WEIGHTS[emotion]
                initial_score = importance * emotional_weight
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "session_id": self.session_id,
                            "pair_id": pair_id,
                            "text": memory["text"],
                            "created_at": now,
                            "last_accessed": now,
                            "access_count": 0,
                            "stability": DEFAULT_STABILITY,
                            "importance": importance,
                            "emotion": emotion,
                            "emotional_weight": emotional_weight,
                            "decay_score": 1.0,
                            "final_score": initial_score,
                            "last_similarity": None,
                        },
                    )
                )
            self._client.upsert(collection_name=self.collection, points=points)

        self._remember_pair(pair_id=pair_id, prompt=prompt, response=response, created_at=now)

    def retrieve(self, prompt: str) -> dict:
        """Return decay-ranked Qdrant memories plus recent prompt/response pairs."""
        memories: list[dict[str, Any]] = []
        if prompt and self._collection_exists():
            vector = self._embed(prompt)
            results = self._search(vector, limit=RETRIEVE_LIMIT)
            reranked = self._rerank_hits(results)
            for hit, payload in reranked[:RETURN_MEMORY_LIMIT]:
                memories.append(
                    {
                        "text": payload.get("text"),
                        "score": payload.get("final_score"),
                        "created_at": payload.get("created_at"),
                        "decay_score": payload.get("decay_score"),
                        "importance": payload.get("importance"),
                        "emotion": payload.get("emotion"),
                    }
                )

        return {
            "recent_pairs": self._recent_pairs(),
            "memories": memories,
        }

    def _extract_memories(self, prompt: str) -> list[dict[str, Any]]:
        llm_response = self.llm.invoke(EXTRACTION_PROMPT.format(prompt=prompt.strip()))
        raw = self._response_text(llm_response).strip()
        if not raw:
            return []

        try:
            parsed = json.loads(raw)
            items = parsed.get("memories", [])
        except json.JSONDecodeError:
            items = [{"text": line.strip("-* \t"), "importance": 0.7, "emotion": "neutral"} for line in raw.splitlines() if line.strip()]

        cleaned = []
        seen = set()
        for item in items:
            if isinstance(item, str):
                item = {"text": item, "importance": 0.7, "emotion": "neutral"}
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(
                {
                    "text": text,
                    "importance": self._normalize_importance(item.get("importance")),
                    "emotion": self._normalize_emotion(item.get("emotion")),
                }
            )
        return cleaned

    def _rerank_hits(self, hits: list[Any]) -> list[tuple[Any, dict[str, Any]]]:
        now = time.time()
        reranked: list[tuple[Any, dict[str, Any]]] = []
        for hit in hits:
            payload = dict(getattr(hit, "payload", None) or {})
            text = payload.get("text")
            if not text:
                continue
            similarity = float(getattr(hit, "score", 0.0) or 0.0)
            updated = self._update_memory_state(payload, similarity=similarity, now=now)
            reranked.append((hit, updated))
            self._set_payload(hit.id, updated)

        reranked.sort(key=lambda item: item[1].get("final_score", 0.0), reverse=True)
        return reranked

    def _update_memory_state(
        self,
        payload: dict[str, Any],
        similarity: float,
        now: float,
    ) -> dict[str, Any]:
        created_at = float(payload.get("created_at", now))
        last_accessed = float(payload.get("last_accessed", created_at))
        access_count = int(payload.get("access_count", 0))
        stability = float(payload.get("stability", DEFAULT_STABILITY))
        importance = self._normalize_importance(payload.get("importance"))
        emotion = self._normalize_emotion(payload.get("emotion"))
        emotional_weight = EMOTION_WEIGHTS[emotion]

        elapsed_days = max((now - last_accessed) / SECONDS_PER_DAY, 0.0)
        decay_score = math.exp(-(elapsed_days / max(stability, 0.1)))
        final_score = similarity * decay_score * importance * emotional_weight

        new_access_count = access_count + 1
        new_stability = stability + (STABILITY_GROWTH * emotional_weight)

        payload.update(
            {
                "created_at": created_at,
                "last_accessed": now,
                "access_count": new_access_count,
                "stability": new_stability,
                "importance": importance,
                "emotion": emotion,
                "emotional_weight": emotional_weight,
                "decay_score": decay_score,
                "final_score": final_score,
                "last_similarity": similarity,
            }
        )
        return payload

    def _remember_pair(self, pair_id: str, prompt: str, response: str, created_at: float) -> None:
        session_pairs = self._session_pairs.setdefault(self.session_id, [])
        session_pairs.append(
            {
                "pair_id": pair_id,
                "prompt": prompt,
                "response": response,
                "created_at": created_at,
            }
        )
        if len(session_pairs) <= SESSION_PAIR_CAP:
            return

        evicted = session_pairs.pop(0)
        self._delete_pair_memories(evicted["pair_id"])

    def _delete_pair_memories(self, pair_id: str) -> None:
        if not self._collection_exists():
            return
        scroll_filter = self._pair_filter(pair_id)
        try:
            points, _ = self._client.scroll(
                collection_name=self.collection,
                scroll_filter=scroll_filter,
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )
        except TypeError:
            points, _ = self._client.scroll(
                collection_name=self.collection,
                scroll_filter=None,
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )
            points = self._filter_points(points, pair_id=pair_id)
        if not points:
            return
        self._client.delete(
            collection_name=self.collection,
            points_selector=PointIdsList(points=[point.id for point in points]),
        )

    def _recent_pairs(self) -> list[dict]:
        pairs = self._session_pairs.get(self.session_id, [])
        recent = pairs[-RECENT_PAIR_LIMIT:]
        return [
            {
                "prompt": item["prompt"],
                "response": item["response"],
                "created_at": item["created_at"],
            }
            for item in recent
        ]

    def _embed(self, text: str) -> list[float]:
        embedded = self._embedder.encode(text, convert_to_numpy=False, normalize_embeddings=True)
        vector = list(embedded)
        if not vector:
            raise ValueError("SentenceTransformer returned an empty vector")
        return [float(value) for value in vector]

    def _ensure_collection(self, vector_size: int) -> None:
        if self._collection_ready:
            return
        if not self._collection_exists():
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        self._collection_ready = True
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        if self._payload_indexes_ready:
            return
        if hasattr(self._client, "create_payload_index"):
            try:
                self._client.create_payload_index(
                    collection_name=self.collection,
                    field_name="session_id",
                    field_schema="keyword",
                )
                self._client.create_payload_index(
                    collection_name=self.collection,
                    field_name="pair_id",
                    field_schema="keyword",
                )
            except UnexpectedResponse as exc:
                # Qdrant returns 400 if the index already exists on a
                # collection created in an earlier run - that's fine, not
                # an error we need to surface.
                if "already exists" not in str(exc):
                    raise
            except Exception as exc:
                print(f"[MemoryOS] Warning: failed to create payload index: {exc}")
        self._payload_indexes_ready = True

    def _collection_exists(self) -> bool:
        try:
            return self._client.collection_exists(self.collection)
        except AttributeError:
            collections = self._client.get_collections().collections
            return any(collection.name == self.collection for collection in collections)
        except UnexpectedResponse:
            return False

    def _session_filter(self):
        return Filter(
            must=[
                FieldCondition(key="session_id", match=MatchValue(value=self.session_id)),
            ]
        )

    def _pair_filter(self, pair_id: str):
        return Filter(
            must=[
                FieldCondition(key="session_id", match=MatchValue(value=self.session_id)),
                FieldCondition(key="pair_id", match=MatchValue(value=pair_id)),
            ]
        )

    def _search(self, vector: list[float], limit: int):
        query_filter = self._session_filter()
        if hasattr(self._client, "query_points"):
            try:
                result = self._client.query_points(
                    collection_name=self.collection,
                    query=vector,
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                )
                return getattr(result, "points", result)
            except TypeError:
                result = self._client.query_points(
                    collection_name=self.collection,
                    query=vector,
                    limit=limit,
                    with_payload=True,
                )
                points = getattr(result, "points", result)
        else:
            try:
                return self._client.search(
                    collection_name=self.collection,
                    query_vector=vector,
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                )
            except TypeError:
                points = self._client.search(
                    collection_name=self.collection,
                    query_vector=vector,
                    limit=limit,
                    with_payload=True,
                )
        return self._filter_points(points)

    def _filter_points(self, points: list[Any], pair_id: str | None = None) -> list[Any]:
        filtered = []
        for point in points:
            payload = getattr(point, "payload", None) or {}
            if payload.get("session_id") != self.session_id:
                continue
            if pair_id is not None and payload.get("pair_id") != pair_id:
                continue
            filtered.append(point)
        return filtered

    def _set_payload(self, point_id: str, payload: dict[str, Any]) -> None:
        if hasattr(self._client, "set_payload"):
            self._client.set_payload(
                collection_name=self.collection,
                payload=payload,
                points=[point_id],
            )
            return

        if hasattr(self._client, "points"):
            for point in self._client.points:
                if point.id == point_id:
                    point.payload.update(payload)
                    return

    @staticmethod
    def _normalize_importance(value: Any) -> float:
        try:
            importance = float(value)
        except (TypeError, ValueError):
            importance = 0.7
        return max(0.1, min(1.0, importance))

    @staticmethod
    def _normalize_emotion(value: Any) -> str:
        emotion = str(value or "neutral").strip().lower()
        return emotion if emotion in EMOTION_WEIGHTS else "neutral"

    @staticmethod
    def _response_text(response: Any) -> str:
        if response is None:
            return ""
        if isinstance(response, str):
            return response
        if hasattr(response, "content"):
            return str(response.content)
        if isinstance(response, dict):
            return str(response.get("content", response.get("text", "")))
        return str(response)