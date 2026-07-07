"""
memory_os/worker.py — Minimal background worker for embedding and upsert

Usage:
    w = Worker(store, embed_model, embed_api_key, redis_url=os.getenv('REDIS_URL'))
    w.enqueue(text, payload)
    # Run worker loop in a background process: w.run()

This is intentionally small and dependency-light. It uses Redis list as a queue
if `redis` is available and `REDIS_URL` is set, otherwise it provides an in-memory
queue useful for tests or single-process deployments.
"""
import os
import time
import json
from typing import Optional

try:
    import litellm
except Exception:
    from . import _litellm as litellm

REDIS_URL = os.getenv("REDIS_URL")
try:
    if REDIS_URL:
        import redis
    else:
        redis = None
except Exception:
    redis = None


class InMemoryQueue:
    def __init__(self):
        self._q = []

    def enqueue(self, item: dict):
        self._q.append(item)

    def dequeue(self) -> Optional[dict]:
        return self._q.pop(0) if self._q else None


class RedisQueue:
    def __init__(self, url: str, key: str = "memory:embedding:queue"):
        if redis is None:
            raise RuntimeError("redis package not available")
        self._r = redis.from_url(url)
        self.key = key

    def enqueue(self, item: dict):
        self._r.rpush(self.key, json.dumps(item))

    def dequeue(self) -> Optional[dict]:
        row = self._r.lpop(self.key)
        if not row:
            return None
        return json.loads(row)


class Worker:
    def __init__(self, store, embed_model: str, embed_api_key: str, redis_url: Optional[str] = None):
        self.store = store
        self.embed_model = embed_model
        self.embed_api_key = embed_api_key
        self.queue = RedisQueue(redis_url) if (redis_url and redis is not None) else InMemoryQueue()

    def enqueue(self, text: str, metadata: dict):
        self.queue.enqueue({"text": text, "metadata": metadata})

    def _embed(self, text: str):
        resp = litellm.embedding(model=self.embed_model, input=[text])
        return resp.data[0]["embedding"]

    def run(self, poll_interval: float = 1.0):
        """Run forever, processing queue items. Intended to be run in background."""
        while True:
            item = self.queue.dequeue()
            if not item:
                time.sleep(poll_interval)
                continue
            text = item.get("text")
            metadata = item.get("metadata", {})
            try:
                vector = self._embed(text)
                # upsert to store — store is expected to have an `upsert_vector` method
                if hasattr(self.store, "upsert_vector"):
                    self.store.upsert_vector(vector=vector, payload=metadata)
                else:
                    # fallback: insert with embedding as extra_payload
                    extra = metadata.copy()
                    extra["_embedding_computed"] = True
                    self.store.insert(text=text, extra_payload=extra)
            except Exception:
                # swallow errors and continue
                continue


# convenience factory
def make_worker(store, embed_model: str, embed_api_key: str):
    return Worker(store=store, embed_model=embed_model, embed_api_key=embed_api_key, redis_url=REDIS_URL)
