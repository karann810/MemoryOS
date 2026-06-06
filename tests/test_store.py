"""
tests/test_store.py — MemoryStore tests
"""
import time
import pytest
from memory_os.store import MemoryStore


@pytest.fixture(scope="module")
def store():
    s = MemoryStore(collection="test_memories")
    yield s
    s.wipe()


def test_insert_returns_uuid(store):
    mid = store.insert("Test memory.")
    assert len(mid) == 36


def test_retrieve_finds_relevant(store):
    store.insert("Paris is the capital of France.", importance=0.9)
    results = store.retrieve("capital of France", top_k=5)
    assert any("Paris" in r.payload["text"] for r in results)


def test_memory_card_fields(store):
    store.insert("Payload check memory.", importance=0.7)
    results = store.retrieve("Payload check", top_k=1)
    p = results[0].payload
    for field in ["text", "importance", "emotional_score", "emotional_label",
                  "created_at", "last_accessed", "access_count",
                  "access_history", "memory_type", "source"]:
        assert field in p, f"Missing field: {field}"


def test_access_history_grows(store):
    store.insert("Access history test.")
    r1 = store.retrieve("Access history test", top_k=3)
    m = next(r for r in r1 if "Access history" in r.payload["text"])
    count1 = len(m.payload["access_history"])

    r2 = store.retrieve("Access history test", top_k=3)
    m2 = next(r for r in r2 if "Access history" in r.payload["text"])
    assert len(m2.payload["access_history"]) == count1 + 1


def test_filter_by_type(store):
    store.insert("Semantic fact.", memory_type="semantic")
    store.insert("Episodic event.", memory_type="episodic")
    results = store.retrieve("fact", top_k=10, memory_type="semantic")
    assert all(r.payload["memory_type"] == "semantic" for r in results)


def test_delete(store):
    mid = store.insert("Delete me.")
    before = store.count()
    store.delete(mid)
    assert store.count() == before - 1
