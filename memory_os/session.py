"""
memory_os/session.py — Redis-backed session cache for working memory
Provides a small abstraction: if REDIS_URL is set and redis is installed,
use Redis for per-user working memory; otherwise fall back to in-process buffer.
"""
from collections import deque
import os
import time
from typing import List, Dict, Optional

REDIS_URL = os.getenv("REDIS_URL")

try:
    if REDIS_URL:
        import redis
    else:
        redis = None
except Exception:
    redis = None


class InMemorySessionStore:
    def __init__(self, maxlen: int = 20):
        self.maxlen = maxlen
        self._buffers: Dict[str, deque] = {}

    def push(self, user_id: str, text: str, role: str = "user") -> None:
        buf = self._buffers.setdefault(user_id, deque(maxlen=self.maxlen))
        buf.append({"text": text, "role": role, "ts": time.time()})

    def retrieve(self, user_id: str, top_k: int = 5) -> List[Dict]:
        buf = list(self._buffers.get(user_id, []))
        recent = buf[-top_k:]
        return [{"text": m["text"], "score": 0.5, "layer": "working"} for m in reversed(recent)]

    def recent_messages(self, user_id: str, top_k: int = 20) -> List[Dict]:
        """Return raw message dicts (most recent first) for pairing logic."""
        buf = list(self._buffers.get(user_id, []))
        recent = buf[-top_k:]
        return list(reversed(recent))

    def flush(self, user_id: str) -> None:
        if user_id in self._buffers:
            del self._buffers[user_id]


class RedisSessionStore:
    """Simple Redis list-based session store.

    Stores JSON-serializable messages per key `session:<user_id>` using LPUSH
    and LRANGE to retrieve most recent messages.
    """
    def __init__(self, redis_url: str, maxlen: int = 20):
        if redis is None:
            raise RuntimeError("redis package not available")
        self._r = redis.from_url(redis_url)
        self.maxlen = maxlen

    def _key(self, user_id: str) -> str:
        return f"session:{user_id}:working"

    def push(self, user_id: str, text: str, role: str = "user") -> None:
        k = self._key(user_id)
        payload = {"text": text, "role": role, "ts": time.time()}
        self._r.lpush(k, repr(payload))
        self._r.ltrim(k, 0, self.maxlen - 1)

    def retrieve(self, user_id: str, top_k: int = 5) -> List[Dict]:
        k = self._key(user_id)
        rows = self._r.lrange(k, 0, top_k - 1)
        results = []
        for b in rows:
            try:
                v = eval(b)
            except Exception:
                continue
            results.append({"text": v.get("text", ""), "score": 0.5, "layer": "working"})
        return results

    def recent_messages(self, user_id: str, top_k: int = 20) -> List[Dict]:
        """Return raw message dicts (most recent first) for pairing logic."""
        k = self._key(user_id)
        rows = self._r.lrange(k, 0, top_k - 1)
        results = []
        for b in rows:
            try:
                v = eval(b)
            except Exception:
                continue
            results.append({"text": v.get("text", ""), "role": v.get("role", "user"), "ts": v.get("ts")})
        return results

    def flush(self, user_id: str) -> None:
        k = self._key(user_id)
        self._r.delete(k)


def get_session_store(maxlen: int = 20, redis_url: Optional[str] = None):
    """Return a session store.

    If `redis_url` is provided it will be used; otherwise the environment
    variable `REDIS_URL` is consulted. If Redis is not available we fall back
    to an in-process store (per-process, per-user deques).
    """
    url = redis_url or REDIS_URL
    if url and redis is not None:
        return RedisSessionStore(url, maxlen=maxlen)
    return InMemorySessionStore(maxlen=maxlen)
