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
    from sklearn.cluster import AgglomerativeClustering
except Exception:
    AgglomerativeClustering = None

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

CONSOLIDATION_DISTANCE_THRESHOLD = 0.15
CONSOLIDATION_TRIGGER_COUNT = 20

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

CONSOLIDATION_PROMPT = """You merge several memory texts into one concise summary.

Preserve every distinct fact from the list below, but return only one clear
plain-text memory item that captures the important details without JSON.

Memory texts:
{memory_texts}
"""


class MemoryOS:
    """Small public API: store prompt memories, retrieve context."""

    _user_pairs: dict[str, list[dict[str, Any]]] = {}

    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        llm: Any,
    ) -> None:
        missing = [
            name
            for name, value in {
                "qdrant_url": qdrant_url,
                "llm": llm,
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
        self.collection = COLLECTION_NAME
        self._client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None)
        self._collection_ready = False
        self._payload_indexes_ready = False
        self._embedder = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)

        self._user_pairs: dict[str, list[dict[str, Any]]] = self._user_pairs
        self._store_counts: dict[str, int] = {}

        # Eagerly create the collection + payload indexes now, instead of
        # waiting for the first successful .store() call. This guarantees
        # .retrieve() never hits a "missing index" 400 error, even if it's
        # called before any .store(), or in a fresh process/kernel.

        vector_size = self._embedder.get_sentence_embedding_dimension()
        self._ensure_collection(vector_size)

    def store(
        self,
        prompt: str,
        response: str,
        user_id: str | None = None,
    ) -> dict[str, bool]:
        """Extract prompt memories into Qdrant and keep the raw pair in user-scoped history. Returns recommendation signal for consolidation."""
        if not prompt or not response:
            return {"consolidation_recommended": False}

        effective_user_id = user_id or ""

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
                            "user_id": effective_user_id,
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

        self._remember_pair(
            pair_id=pair_id,
            prompt=prompt,
            response=response,
            created_at=now,
            user_id=effective_user_id,
        )

        current_count = self._store_counts.get(effective_user_id, 0) + 1
        if current_count >= CONSOLIDATION_TRIGGER_COUNT:
            self._store_counts[effective_user_id] = 0
            recommended = True
        else:
            self._store_counts[effective_user_id] = current_count
            recommended = False

        return {"consolidation_recommended": recommended}

    def retrieve(self, prompt: str, user_id: str | None = None) -> dict:
        """Return decay-ranked Qdrant memories plus recent prompt/response pairs.
        Optional `user_id` scopes the read to a distinct user memory namespace.
        """
        memories: list[dict[str, Any]] = []
        effective_user_id = user_id or ""
        if prompt and self._collection_exists():
            vector = self._embed(prompt)
            results = self._search(
                vector,
                limit=RETRIEVE_LIMIT,
                user_id=effective_user_id,
            )
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
            "recent_pairs": self._recent_pairs(effective_user_id),
            "memories": memories,
        }

    def consolidate(self, user_id: str) -> dict:
        """Cluster and merge semantically-similar user memories into one summary point."""
        self._store_counts[user_id] = 0
        if not self._collection_exists():
            return {"clusters_merged": 0, "points_removed": 0}

        points = self._scroll_user_points_with_vectors(user_id)
        if len(points) < 2:
            return {"clusters_merged": 0, "points_removed": 0}

        if AgglomerativeClustering is None:
            raise ImportError(
                "MemoryOS consolidation requires scikit-learn. Install package dependencies before use."
            )

        vectors = [self._point_vector(point) for point in points]
        if not vectors:
            return {"clusters_merged": 0, "points_removed": 0}

        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=CONSOLIDATION_DISTANCE_THRESHOLD,
            linkage="average",
            metric="cosine",
        ).fit_predict(vectors)

        groups: dict[int, list[Any]] = {}
        for point, label in zip(points, labels):
            groups.setdefault(int(label), []).append(point)

        clusters_merged = 0
        points_removed = 0
        for cluster in groups.values():
            if len(cluster) == 1:
                continue

            texts = [self._point_text(point) for point in cluster]
            merged_text = self._response_text(
                self.llm.invoke(CONSOLIDATION_PROMPT.format(memory_texts="\n".join(texts)))
            ).strip()
            if not merged_text:
                merged_text = " ".join(texts)

            merged_vector = self._embed(merged_text)
            highest_importance_point = max(
                cluster,
                key=lambda point: float((getattr(point, "payload", None) or {}).get("importance", 0.0)),
            )
            highest_importance_payload = getattr(highest_importance_point, "payload", None) or {}
            emotion = str(highest_importance_payload.get("emotion", "neutral")).strip().lower()
            if emotion not in EMOTION_WEIGHTS:
                emotion = "neutral"

            merged_importance = max(
                float((getattr(point, "payload", None) or {}).get("importance", 0.0))
                for point in cluster
            )
            merged_emotional_weight = EMOTION_WEIGHTS[emotion]

            merged_payload = {
                "user_id": user_id,
                "text": merged_text,
                "importance": merged_importance,
                "access_count": sum(
                    int((getattr(point, "payload", None) or {}).get("access_count", 0))
                    for point in cluster
                ),
                "created_at": min(
                    float((getattr(point, "payload", None) or {}).get("created_at", time.time()))
                    for point in cluster
                ),
                "last_accessed": max(
                    float((getattr(point, "payload", None) or {}).get("last_accessed", time.time()))
                    for point in cluster
                ),
                "emotion": emotion,
                "emotional_weight": merged_emotional_weight,
                "stability": DEFAULT_STABILITY,
                "decay_score": 1.0,
                "final_score": merged_importance * merged_emotional_weight,
                "pair_id": None,
                # Merged points intentionally fall outside the pair-based eviction lifecycle
                # because they are no longer tied to one prompt/response pair.
                "merged_from": len(cluster),
            }

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=merged_vector,
                payload=merged_payload,
            )
            self._client.upsert(collection_name=self.collection, points=[point])
            self._client.delete(
                collection_name=self.collection,
                points_selector=PointIdsList(points=[original.id for original in cluster]),
            )

            clusters_merged += 1
            points_removed += len(cluster)

        return {"clusters_merged": clusters_merged, "points_removed": points_removed}

    def _scroll_user_points_with_vectors(self, user_id: str) -> list[Any]:
        if not self._collection_exists():
            return []

        scroll_filter = self._user_filter(user_id)
        try:
            points, _ = self._client.scroll(
                collection_name=self.collection,
                scroll_filter=scroll_filter,
                limit=1000,
                with_payload=True,
                with_vectors=True,
            )
        except TypeError:
            points, _ = self._client.scroll(
                collection_name=self.collection,
                scroll_filter=None,
                limit=1000,
                with_payload=True,
                with_vectors=True,
            )
            points = [point for point in points if getattr(point, "payload", {}).get("user_id") == user_id]
        return [point for point in points if getattr(point, "payload", {}).get("user_id") == user_id]

    @staticmethod
    def _point_text(point: Any) -> str:
        payload = getattr(point, "payload", None) or {}
        return str(payload.get("text", "")).strip()

    @staticmethod
    def _point_vector(point: Any) -> list[float]:
        vector = getattr(point, "vector", None)
        if vector is None:
            vector = getattr(point, "vectors", None)
        if isinstance(vector, dict):
            for value in vector.values():
                if isinstance(value, list):
                    vector = value
                    break
        if isinstance(vector, list):
            return [float(value) for value in vector]
        return []

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

    def _remember_pair(
        self,
        pair_id: str,
        prompt: str,
        response: str,
        created_at: float,
        user_id: str | None = None,
    ) -> None:
        effective_user_id = user_id or ""
        user_pairs = self._user_pairs.setdefault(effective_user_id, [])
        user_pairs.append(
            {
                "pair_id": pair_id,
                "prompt": prompt,
                "response": response,
                "created_at": created_at,
                "user_id": effective_user_id,
            }
        )
        if len(user_pairs) <= SESSION_PAIR_CAP:
            return

        evicted = user_pairs.pop(0)
        self._delete_pair_memories(effective_user_id, evicted["pair_id"])

    def _delete_pair_memories(self, user_id: str, pair_id: str) -> None:
        if not self._collection_exists():
            return
        scroll_filter = self._pair_filter(user_id, pair_id)
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
            points = self._filter_points(points, pair_id=pair_id, user_id=user_id)
        if not points:
            return
        self._client.delete(
            collection_name=self.collection,
            points_selector=PointIdsList(points=[point.id for point in points]),
        )

    def _recent_pairs(self, user_id: str | None = None) -> list[dict]:
        effective_user_id = user_id or ""
        pairs = self._user_pairs.get(effective_user_id, [])
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
                    field_name="user_id",
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

#done just returns true if collection exists, false if not. If the collection does not exist, 
# we will create it in _ensure_collection

    def _collection_exists(self) -> bool:
        try:
            return self._client.collection_exists(self.collection)
        except AttributeError:
            collections = self._client.get_collections().collections
            return any(collection.name == self.collection for collection in collections)
        except UnexpectedResponse:
            return False

    def _user_filter(self, user_id: str):
        return Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            ]
        )

    def _pair_filter(self, user_id: str, pair_id: str):
        return Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="pair_id", match=MatchValue(value=pair_id)),
            ]
        )

    def _search(self, vector: list[float], limit: int, user_id: str | None = None) -> list[Any]:
        query_filter = self._user_filter(user_id or "")
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
            except TypeError as e:
        # 2. DO NOT fall back to a filterless search. 
        # Instead, raise a secure error so you can see exactly what is wrong with the filter format.
                raise ValueError(
            f"CRITICAL: Query filter type mismatch. To prevent multi-user leak, "
            f"search was blocked. Original error: {e}"
        ) from e
        else:
            try:
                return self._client.search(
                    collection_name=self.collection,
                    query_vector=vector,
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                )
            except TypeError as e:
                raise ValueError(
                    f"CRITICAL: Search filter type mismatch. To prevent multi-user leak, "
                    f"search was blocked. Original error: {e}"
                ) from e
        # return self._filter_points(points, user_id=user_id)

    def _filter_points(
        self,
        points: list[Any],
        pair_id: str | None = None,
        user_id: str | None = None,
    ) -> list[Any]:
        effective_user_id = user_id or ""
        filtered = []
        for point in points:
            payload = getattr(point, "payload", None) or {}
            if payload.get("user_id") != effective_user_id:
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

    # @staticmethod
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