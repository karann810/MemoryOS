"""
memory_os/router.py  —  Week 4+6+7: 3-Layer Memory Router
==========================================================
Routes queries to the right memory layer:

  WORKING memory   → Pinecone (serverless, ultra-low latency)
    - Last 15 messages
    - Flushed per session
    - Fetched on EVERY turn

  EPISODIC memory  → Qdrant (time-scored, decay-weighted)
    - Specific events with timestamps
    - Decays via Ebbinghaus curve
    - Queried for relevant past events

  SEMANTIC memory  → Weaviate (hybrid BM25 + vector search)
    - Compressed summaries / facts / preferences
    - Searched by keyword AND meaning
    - Long-lived, high importance

The router decides which layer(s) to query based on question type,
then merges and re-ranks results across layers.
"""

import os
import time
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


# ─── Weaviate client (semantic layer) ────────────────────────────────────────

def _get_weaviate_client():
    try:
        import weaviate
        return weaviate.connect_to_wcs(
            cluster_url=os.getenv("WEAVIATE_URL", ""),
            auth_credentials=weaviate.auth.AuthApiKey(
                os.getenv("WEAVIATE_API_KEY", "")
            ),
        )
    except Exception as e:
        raise RuntimeError(f"Weaviate connection failed: {e}. "
                           "Set WEAVIATE_URL and WEAVIATE_API_KEY in .env")


# ─── Pinecone client (working layer) ─────────────────────────────────────────

def _get_pinecone_index():
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", ""))
        index_name = os.getenv("PINECONE_INDEX", "working-memory")
        return pc.Index(index_name)
    except Exception as e:
        raise RuntimeError(f"Pinecone connection failed: {e}. "
                           "Set PINECONE_API_KEY and PINECONE_INDEX in .env")


# ─── Query classifier ─────────────────────────────────────────────────────────

QUESTION_PATTERNS = {
    "working":  [
        "what did i just", "what was i saying", "earlier in this",
        "just mentioned", "you said", "we were talking",
    ],
    "semantic": [
        "what do i prefer", "do i like", "what's my", "my preference",
        "how do i usually", "do i always", "what kind of person",
        "what are my", "tell me about myself",
    ],
    "episodic": [
        "when did", "what happened", "last time", "remember when",
        "did i ever", "have i", "what did i do",
    ],
}


def classify_query(query: str) -> list[str]:
    """
    Classify query → list of memory layers to search.
    Returns one or more of: ["working", "episodic", "semantic"]
    Most queries search episodic + semantic; working is always included.
    """
    q = query.lower()
    layers = {"working"}  # always check working memory

    for layer, patterns in QUESTION_PATTERNS.items():
        if any(p in q for p in patterns):
            layers.add(layer)

    # Default: search all three if no specific pattern matched
    if len(layers) == 1:
        layers = {"working", "episodic", "semantic"}

    return list(layers)


# ─── Working memory (Pinecone) ────────────────────────────────────────────────

class WorkingMemory:
    """
    Last 15 messages, stored in Pinecone serverless.
    Flushed at session end.
    """

    def __init__(self, session_id: str, embed_fn):
        self._index      = _get_pinecone_index()
        self._session_id = session_id
        self._embed      = embed_fn

    def push(self, text: str, role: str = "user") -> None:
        """Add a message to working memory."""
        vec = self._embed(text)
        msg_id = f"{self._session_id}_{int(time.time() * 1000)}"
        self._index.upsert(vectors=[{
            "id":     msg_id,
            "values": vec,
            "metadata": {
                "text":       text,
                "role":       role,
                "session_id": self._session_id,
                "timestamp":  time.time(),
            }
        }], namespace=self._session_id)
        self._prune()

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Fetch most relevant recent messages."""
        vec = self._embed(query)
        results = self._index.query(
            vector=vec,
            top_k=top_k,
            namespace=self._session_id,
            include_metadata=True,
        )
        return [
            {"text": m.metadata["text"], "score": m.score, "layer": "working"}
            for m in results.matches
        ]

    def flush(self) -> None:
        """Clear working memory at session end."""
        self._index.delete(delete_all=True, namespace=self._session_id)

    def _prune(self) -> None:
        """Keep only the 15 most recent messages."""
        results = self._index.query(
            vector=[0.0] * int(os.getenv("VECTOR_DIM", "1536")),
            top_k=100,
            namespace=self._session_id,
            include_metadata=True,
        )
        if len(results.matches) > 15:
            oldest = sorted(results.matches, key=lambda m: m.metadata.get("timestamp", 0))
            to_delete = [m.id for m in oldest[:-15]]
            self._index.delete(ids=to_delete, namespace=self._session_id)


# ─── Semantic memory (Weaviate) ───────────────────────────────────────────────

class SemanticMemory:
    """
    Compressed semantic facts in Weaviate with hybrid BM25 + vector search.
    """
    CLASS_NAME = "SemanticMemory"

    def __init__(self, embed_fn):
        self._client = _get_weaviate_client()
        self._embed  = embed_fn
        self._ensure_schema()

    def _ensure_schema(self):
        try:
            self._client.collections.get(self.CLASS_NAME)
        except Exception:
            self._client.collections.create(
                name=self.CLASS_NAME,
                properties=[
                    {"name": "text",            "dataType": ["text"]},
                    {"name": "importance",      "dataType": ["number"]},
                    {"name": "emotional_label", "dataType": ["text"]},
                    {"name": "emotional_score", "dataType": ["number"]},
                    {"name": "source_count",    "dataType": ["int"]},
                    {"name": "created_at",      "dataType": ["number"]},
                ],
            )

    def insert(self, text: str, payload: dict) -> str:
        """Store a semantic memory."""
        collection = self._client.collections.get(self.CLASS_NAME)
        result = collection.data.insert({
            "text":            text,
            "importance":      payload.get("importance", 0.5),
            "emotional_label": payload.get("emotional_label", "neutral"),
            "emotional_score": payload.get("emotional_score", 0.0),
            "source_count":    payload.get("source_count", 1),
            "created_at":      time.time(),
        })
        return str(result)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Hybrid search: BM25 keyword + vector similarity."""
        collection = self._client.collections.get(self.CLASS_NAME)
        results = collection.query.hybrid(
            query=query,
            limit=top_k,
            return_metadata=["score"],
        )
        return [
            {
                "text":  obj.properties.get("text", ""),
                "score": obj.metadata.score if obj.metadata else 0.0,
                "layer": "semantic",
                "payload": obj.properties,
            }
            for obj in results.objects
        ]


# ─── Main Router ──────────────────────────────────────────────────────────────

class MemoryRouter:
    """
    Routes queries across all 3 memory layers and merges results.

    Usage:
        router = MemoryRouter(store, session_id="user_123")
        results = router.query("What does the user prefer?", top_n=5)
        for r in results:
            print(r["layer"], r["text"])
    """

    def __init__(
        self,
        store,                            # MemoryStore (Qdrant episodic)
        decay_reranker=None,              # DecayReranker instance
        session_id: str = "default",
        embed_fn=None,                    # pass store's embed fn or custom
        use_pinecone: bool = True,
        use_weaviate: bool = True,
    ):
        from .decay import DecayReranker
        self.store          = store
        self.reranker       = decay_reranker or DecayReranker()
        self.session_id     = session_id
        self._embed         = embed_fn or self._default_embed

        self.working  = WorkingMemory(session_id, self._embed) if use_pinecone else None
        self.semantic = SemanticMemory(self._embed) if use_weaviate else None

    def _default_embed(self, text: str) -> list[float]:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            input=text,
        )
        return response.data[0].embedding

    def query(self, query: str, top_n: int = 5) -> list[dict]:
        """
        Route query to appropriate layers and return merged top_n results.
        """
        layers = classify_query(query)
        all_results = []

        if "working" in layers and self.working:
            working_results = self.working.retrieve(query, top_k=5)
            all_results.extend(working_results)

        if "episodic" in layers:
            raw = self.store.retrieve(query, top_k=50, memory_type="episodic")
            ranked = self.reranker.rerank(raw, top_n=10)
            all_results.extend([
                {
                    "text":    m.text,
                    "score":   m.final_score,
                    "layer":   "episodic",
                    "payload": m.payload,
                }
                for m in ranked
            ])

        if "semantic" in layers and self.semantic:
            semantic_results = self.semantic.retrieve(query, top_k=5)
            all_results.extend(semantic_results)

        # Sort by score and deduplicate
        all_results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        seen_texts = set()
        deduped = []
        for r in all_results:
            key = r["text"][:100]
            if key not in seen_texts:
                seen_texts.add(key)
                deduped.append(r)

        return deduped[:top_n]

    def push_to_working(self, text: str, role: str = "user") -> None:
        """Add current message to working memory."""
        if self.working:
            self.working.push(text, role)

    def flush_working(self) -> None:
        """Clear working memory at session end."""
        if self.working:
            self.working.flush()

    def add_semantic(self, text: str, payload: dict) -> None:
        """Directly add to semantic layer (called by consolidator)."""
        if self.semantic:
            self.semantic.insert(text, payload)
