"""
tests/test_emotion.py — EmotionTagger tests (keyword mode, no API needed)
"""
import pytest
from memory_os.emotion import EmotionTagger


@pytest.fixture
def tagger():
    return EmotionTagger(mode="keyword")


def test_neutral_factual_text(tagger):
    result = tagger.tag("Python was released in 1991.")
    assert result["label"] == "neutral"
    assert result["score"] < 0.3


def test_fear_detection(tagger):
    result = tagger.tag("I'm terrified and anxious about the deadline.")
    assert result["label"] == "fear"
    assert result["score"] > 0.0


def test_joy_detection(tagger):
    result = tagger.tag("I'm so happy and excited about this success!")
    assert result["label"] == "joy"


def test_anger_detection(tagger):
    result = tagger.tag("I'm really angry and frustrated with this.")
    assert result["label"] == "anger"


def test_result_has_all_fields(tagger):
    result = tagger.tag("Some text.")
    assert "label"     in result
    assert "score"     in result
    assert "reasoning" in result


def test_score_bounded(tagger):
    result = tagger.tag("I hate this terrible awful thing so much!")
    assert 0.0 <= result["score"] <= 1.0


def test_batch_tag(tagger):
    texts   = ["Hello.", "I'm scared!", "Great success!"]
    results = tagger.batch_tag(texts)
    assert len(results) == 3
    assert all("label" in r for r in results)
