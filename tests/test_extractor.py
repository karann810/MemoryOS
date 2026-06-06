"""
tests/test_extractor.py — MemoryExtractor tests (keyword mode, no API needed)
Uses mock LLM responses to test extraction logic without API calls.
"""
import pytest
from unittest.mock import MagicMock, patch
from memory_os.extractor import MemoryExtractor, _is_trivial


# ── Trivial message detection ─────────────────────────────────────────────────

def test_trivial_hi():
    assert _is_trivial("hi") is True

def test_trivial_thanks():
    assert _is_trivial("thanks") is True

def test_trivial_short():
    assert _is_trivial("ok sure") is True

def test_not_trivial_long():
    assert _is_trivial("I'm building a SaaS with Next.js") is False


# ── Extraction with mocked LLM ────────────────────────────────────────────────

MOCK_EXTRACTION = [
    {
        "text":        "User is building a SaaS product with Next.js",
        "importance":  0.9,
        "memory_type": "semantic",
        "confidence":  0.95,
    },
    {
        "text":        "User hates Firebase",
        "importance":  0.8,
        "memory_type": "semantic",
        "confidence":  0.9,
    },
    {
        "text":        "User is stressed about a 3-week launch deadline",
        "importance":  0.7,
        "memory_type": "episodic",
        "confidence":  0.85,
    },
]

import json

def make_mock_oai(response_json):
    mock = MagicMock()
    mock.chat.completions.create.return_value.choices[0].message.content = (
        json.dumps(response_json)
    )
    return mock


def test_extract_returns_list():
    extractor = MemoryExtractor()
    extractor._oai = make_mock_oai(MOCK_EXTRACTION)
    results = extractor.extract(
        "I'm building a SaaS with Next.js, tried Firebase and hated it, "
        "3 week deadline and super stressed"
    )
    assert isinstance(results, list)
    assert len(results) == 3


def test_extract_has_required_fields():
    extractor = MemoryExtractor()
    extractor._oai = make_mock_oai(MOCK_EXTRACTION)
    results = extractor.extract("I'm building with Next.js and hate Firebase")
    for r in results:
        assert "text"        in r
        assert "importance"  in r
        assert "memory_type" in r
        assert "confidence"  in r


def test_extract_memory_types_valid():
    extractor = MemoryExtractor()
    extractor._oai = make_mock_oai(MOCK_EXTRACTION)
    results = extractor.extract("I'm building with Next.js and hate Firebase")
    for r in results:
        assert r["memory_type"] in ("episodic", "semantic")


def test_extract_filters_low_importance():
    low_importance = [
        {"text": "User said hi", "importance": 0.1,
         "memory_type": "episodic", "confidence": 0.9},
        {"text": "User prefers Python", "importance": 0.9,
         "memory_type": "semantic", "confidence": 0.9},
    ]
    extractor = MemoryExtractor(min_importance=0.3)
    extractor._oai = make_mock_oai(low_importance)
    results = extractor.extract("Hi, I prefer Python")
    # low importance one should be filtered
    assert all(r["importance"] >= 0.3 for r in results)


def test_extract_trivial_message_skips_api():
    extractor = MemoryExtractor()
    mock_oai  = MagicMock()
    extractor._oai = mock_oai
    results = extractor.extract("hi")
    # Should return empty without calling API
    mock_oai.chat.completions.create.assert_not_called()
    assert results == []


def test_extract_returns_empty_on_api_error():
    extractor = MemoryExtractor()
    extractor._oai = MagicMock(
        side_effect=Exception("API error")
    )
    # Should not crash, just return empty
    results = extractor.extract("I'm building something with Next.js")
    assert isinstance(results, list)


def test_separate_memories_for_separate_facts():
    extractor = MemoryExtractor()
    extractor._oai = make_mock_oai(MOCK_EXTRACTION)
    results = extractor.extract(
        "I hate Firebase, love Next.js, stressed about deadline"
    )
    # Should be 3 separate memories, not 1
    assert len(results) == 3
    texts = [r["text"] for r in results]
    assert any("Firebase" in t for t in texts)
    assert any("Next.js"  in t for t in texts)
    assert any("deadline" in t or "stressed" in t for t in texts)
