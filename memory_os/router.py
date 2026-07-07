"""
memory_os/router.py  —  3-Layer Memory Router
==============================================
Routes queries to the right memory layer:

  WORKING memory   → In-memory deque (last 20 messages, current session only)
                     Fast, no extra service needed.

  EPISODIC memory  → Qdrant (time-scored, decay-weighted)
                     Specific events with timestamps, Ebbinghaus decay.

  SEMANTIC memory  → Qdrant (separate collection, high-importance summaries)
                     Written by the consolidator after clustering.

All layers use Qdrant — no Pinecone or Weaviate required.
Embeddings via litellm so any provider works.
"""

import os
import time
from collections import deque
from typing import Optional


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


def classify_query(query: str) -> set[str]:
    q      = query.lower()
    layers = {"episodic"}  # always search episodic
    for layer, patterns in QUESTION_PATTERNS.items():
        if any(p in q for p in patterns):
            layers.add(layer)
    if len(layers) == 1:
        layers = {"working", "episodic", "semantic"}
    layers.add("working")
    return layers


class WorkingMemory:
    """
    In-memory ring buffers keyed by `user_id`.

    This mirrors the session store API so the router can treat the
    working memory uniformly whether it's Redis-backed or in-process.
    """

    def __init__(self, maxlen: int = 20):
        self.maxlen = maxlen
        self._buffers: dict = {}

    def push(self, user_id_or_text, text_or_role=None, role: str = "user") -> None:
        """Support both `push(user_id, text, role)` and `push(text, role)`.

        When a `user_id` is provided we store per-user. If called with
        `(text, role)` we use a default key to preserve backwards compat.
        """
        if text_or_role is None:
            # signature: push(text, role)
            user_id = "default"
            text = user_id_or_text
            r = role
        else:
            user_id = user_id_or_text
            text = text_or_role
            r = role

        buf = self._buffers.setdefault(user_id, deque(maxlen=self.maxlen))
        buf.append({"text": text, "role": r, "ts": time.time()})

    def retrieve(self, user_id: str, top_k: int = 5) -> list[dict]:
        """Return most recent messages for a particular `user_id`."""
        buf = list(self._buffers.get(user_id, []))
        recent = buf[-top_k:]
        return [
            {"text": m["text"], "score": 0.5, "layer": "working"}
            for m in reversed(recent)
        ]

    def recent_messages(self, user_id: str, top_k: int = 20) -> list[dict]:
        """Return raw message dicts (most recent first) for pairing logic."""
        buf = list(self._buffers.get(user_id, []))
        recent = buf[-top_k:]
        return [m for m in reversed(recent)]

    def flush(self, user_id: str | None = None) -> None:
        if user_id is None:
            self._buffers.clear()
        else:
            if user_id in self._buffers:
                del self._buffers[user_id]


class MemoryRouter:
    """
    Routes queries across memory layers, merges and deduplicates results.

    Usage:
        router = MemoryRouter(store, session_id="user_123",
                              embed_model="text-embedding-3-small",
                              embed_api_key="sk-...")
        results = router.query("What does the user prefer?", top_n=5)
    """

    def __init__(
        self,
        store,
        decay_reranker=None,
        session_id:    str = "default",
        embed_model:   str = "text-embedding-3-small",
        embed_api_key: str = "",
        redis_url:     Optional[str] = None,
    ):
        from .decay import DecayReranker
        self.store      = store
        self.reranker   = decay_reranker or DecayReranker()
        self.session_id = session_id
        # Use Redis-backed session cache when available, otherwise local deque
        try:
            from .session import get_session_store
            self.working = get_session_store(maxlen=20, redis_url=redis_url)
        except Exception:
            self.working = WorkingMemory(maxlen=20)

    def query(self, query: str, top_n: int = 5) -> list[dict]:
        """Route query across layers, return merged top_n results."""
        layers      = classify_query(query)
        all_results = []

        if "working" in layers:
            # Ask the session-backed working store for this session's recent messages
            try:
                all_results.extend(self.working.retrieve(self.session_id, top_k=5))
            except TypeError:
                # Fallback if a legacy WorkingMemory is present
                all_results.extend(self.working.retrieve(query, top_k=5))

        if "episodic" in layers:
            raw    = self.store.retrieve(query, top_k=50, memory_type="episodic", user_id=self.session_id)
            ranked = self.reranker.rerank(raw, top_n=15)
            all_results.extend([
                {"text": m.text, "score": m.final_score, "layer": "episodic", "payload": m.payload}
                for m in ranked
            ])

        if "semantic" in layers:
            raw    = self.store.retrieve(query, top_k=20, memory_type="semantic", user_id=self.session_id)
            ranked = self.reranker.rerank(raw, top_n=10)
            all_results.extend([
                {"text": m.text, "score": m.final_score, "layer": "semantic", "payload": m.payload}
                for m in ranked
            ])

        # Sort by score, deduplicate
        all_results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        seen, deduped = set(), []
        for r in all_results:
            key = r["text"][:100]
            if key not in seen:
                seen.add(key)
                deduped.append(r)

        return deduped[:top_n]

    def push_to_working(self, text: str, role: str = "user") -> None:
        # Support different session store signatures: some stores expect
        # (user_id, text, role), others expect (text, role).
        try:
            # Redis/InMemory session stores expect user_id first
            self.working.push(self.session_id, text, role)
        except TypeError:
            # WorkingMemory expects (text, role)
            self.working.push(text, role)

    def flush_working(self) -> None:
        self.working.flush()

    def get_recent_pairs(self, n: int = 3) -> list[dict]:
        """Return up to `n` recent (user, assistant) pairs, oldest->newest."""
        # Fetch raw messages; different stores have different signatures
        msgs = []
        try:
            msgs = self.working.recent_messages(self.session_id, top_k=n * 4)
        except TypeError:
            try:
                msgs = self.working.recent_messages(top_k=n * 4)
            except Exception:
                msgs = []

        # msgs: most recent first -> reverse to oldest->newest
        msgs = list(reversed(msgs))
        pairs = []
        last_user = None
        for m in msgs:
            role = m.get("role") or m.get("r") or "user"
            if role == "user":
                last_user = m
            elif role == "assistant" and last_user is not None:
                pairs.append({"user": last_user.get("text"), "assistant": m.get("text"), "ts": m.get("ts")})
                last_user = None
            if len(pairs) >= n:
                break

        return pairs
