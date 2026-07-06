"""
memory_os/router.py  —  Qdrant-only working-memory router
==========================================================
Routes recent prompt/response messages stored in Qdrant by user_id.
"""

import time


class MemoryRouter:
    """
    Routes recent working-memory messages via Qdrant.

    Usage:
        router = MemoryRouter(store, session_id="user_123")
        results = router.query("ignored", top_n=6)
        for item in results:
            print(item["payload"]["role"], item["text"])
    """

    def __init__(self, store, session_id: str = "default"):
        self.store = store
        self.session_id = session_id

    def query(self, query: str, top_n: int = 6) -> list[dict]:
        """Return the last prompt/response pairs for this user."""
        entries = self.store.retrieve_recent(
            user_id=self.session_id,
            top_k=top_n,
            memory_type="working",
        )
        return [
            {
                "text": entry.payload.get("text", ""),
                "layer": entry.payload.get("memory_type", "working"),
                "payload": entry.payload,
            }
            for entry in entries
        ]

    def push_to_working(self, text: str, role: str = "user") -> None:
        """Store a recent conversation message in Qdrant."""
        self.store.insert(
            text=text,
            importance=0.0,
            memory_type="working",
            emotional_score=0.0,
            emotional_label="neutral",
            source=f"session:{self.session_id}",
            extra_payload={
                "user_id": self.session_id,
                "role": role,
                "created_at": time.time(),
            },
        )

    def flush_working(self) -> None:
        """Delete working-memory messages for this user."""
        self.store.delete_by_filter(
            user_id=self.session_id,
            memory_type="working",
        )

    def add_semantic(self, text: str, payload: dict) -> None:
        """No-op for Qdrant-only mode."""
        pass
