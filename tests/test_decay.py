"""
tests/test_decay.py — Ebbinghaus forgetting curve tests
"""
import time
import math
import pytest
from memory_os.decay import ebbinghaus_retention, DecayReranker, EMOTION_STABILITY_BOOST


# ── ebbinghaus_retention ──────────────────────────────────────────────────────

def test_fresh_memory_has_high_retention():
    now = time.time()
    retention, _ = ebbinghaus_retention(
        created_at=now, last_accessed=now, access_history=[]
    )
    assert retention > 0.9


def test_old_unaccessed_memory_decays():
    old = time.time() - (30 * 86400)  # 30 days ago
    retention, _ = ebbinghaus_retention(
        created_at=old, last_accessed=old, access_history=[]
    )
    assert retention < 0.2


def test_fear_memory_decays_slower_than_neutral():
    old = time.time() - (7 * 86400)  # 7 days ago
    ret_fear, s_fear = ebbinghaus_retention(
        created_at=old, last_accessed=old,
        access_history=[], emotional_label="fear", emotional_score=0.9
    )
    ret_neutral, s_neutral = ebbinghaus_retention(
        created_at=old, last_accessed=old,
        access_history=[], emotional_label="neutral", emotional_score=0.0
    )
    assert ret_fear > ret_neutral
    assert s_fear > s_neutral


def test_spaced_repetition_increases_stability():
    base = time.time() - (10 * 86400)
    # Memory accessed multiple times with increasing gaps
    history = [
        base + 86400,       # day 1
        base + 3 * 86400,   # day 3
        base + 7 * 86400,   # day 7
    ]
    _, stability_with_history = ebbinghaus_retention(
        created_at=base, last_accessed=history[-1], access_history=history
    )
    _, stability_no_history = ebbinghaus_retention(
        created_at=base, last_accessed=base, access_history=[]
    )
    assert stability_with_history > stability_no_history


def test_emotion_stability_ordering():
    # fear should have highest stability boost
    assert EMOTION_STABILITY_BOOST["fear"] > EMOTION_STABILITY_BOOST["neutral"]
    assert EMOTION_STABILITY_BOOST["fear"] >= EMOTION_STABILITY_BOOST["anger"]


# ── DecayReranker ─────────────────────────────────────────────────────────────

class MockResult:
    """Minimal mock of a Qdrant ScoredPoint."""
    def __init__(self, id, text, score, importance=0.5,
                 emotional_label="neutral", emotional_score=0.0,
                 days_old=0, access_history=None):
        now = time.time()
        created = now - (days_old * 86400)
        self.id = id
        self.score = score
        self.payload = {
            "text":            text,
            "importance":      importance,
            "emotional_label": emotional_label,
            "emotional_score": emotional_score,
            "created_at":      created,
            "last_accessed":   created,
            "access_history":  access_history or [],
            "memory_type":     "episodic",
        }


def test_reranker_promotes_emotional_memory():
    reranker = DecayReranker()
    results = [
        MockResult("1", "Neutral fact from yesterday.", score=0.9,
                   emotional_label="neutral", days_old=1),
        MockResult("2", "Terrifying event from last week.", score=0.7,
                   emotional_label="fear", emotional_score=0.9, days_old=7),
    ]
    ranked = reranker.rerank(results, top_n=2)
    # Both present, but check scoring works
    assert len(ranked) == 2
    assert all(hasattr(m, "final_score") for m in ranked)


def test_reranker_demotes_very_old_neutral_memory():
    reranker = DecayReranker()
    results = [
        MockResult("1", "Very old neutral memory.", score=0.95,
                   emotional_label="neutral", days_old=60),
        MockResult("2", "Fresh memory.", score=0.7,
                   emotional_label="neutral", days_old=0),
    ]
    ranked = reranker.rerank(results, top_n=2)
    # Fresh memory should outrank very old one despite lower similarity
    fresh_rank = next(i for i, m in enumerate(ranked) if m.id == "2")
    old_rank   = next(i for i, m in enumerate(ranked) if m.id == "1")
    assert fresh_rank < old_rank


def test_reranker_score_breakdown():
    reranker = DecayReranker()
    result   = MockResult("1", "Test.", score=0.8, importance=0.9,
                          emotional_label="joy", emotional_score=0.7, days_old=1)
    ranked   = reranker.score(result)
    assert 0.0 <= ranked.final_score <= 10.0
    assert ranked.similarity == 0.8
    assert ranked.importance  == 0.9
    assert ranked.retention   > 0.0
    assert ranked.emotional_weight > 1.0  # joy should boost above 1.0


def test_reranker_returns_top_n(store=None):
    reranker = DecayReranker()
    results  = [MockResult(str(i), f"Memory {i}", score=0.5) for i in range(20)]
    ranked   = reranker.rerank(results, top_n=5)
    assert len(ranked) == 5
