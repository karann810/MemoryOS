"""
memory_os/emotion.py  —  Emotional Tagger
==========================================
Tags every memory with an emotional label and intensity score
before it gets stored. This feeds directly into the Ebbinghaus
stability calculation in decay.py.

Why this matters:
  McGaugh (2000) showed that emotional arousal during or after
  an event triggers amygdala activation which strengthens
  hippocampal memory consolidation. Fearful/angry/joyful events
  are remembered far better than neutral ones.

  No existing AI memory tool implements this. We do.

Two modes:
  1. LLM mode (default, accurate)  — asks GPT to classify emotion
  2. Keyword mode (fast, offline)  — simple keyword matching
     useful for testing without API calls
"""

import os
import json
import re
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

EMOTION_LABELS = ["joy", "fear", "anger", "sadness", "surprise", "neutral"]

# Keyword fallback for fast/offline mode
EMOTION_KEYWORDS = {
    "joy":      ["happy", "excited", "love", "great", "wonderful", "amazing",
                 "delighted", "pleased", "glad", "celebrate", "win", "success"],
    "fear":     ["scared", "afraid", "terrified", "worried", "anxious",
                 "nervous", "panic", "dread", "threat", "danger", "risk"],
    "anger":    ["angry", "furious", "annoyed", "frustrated", "hate",
                 "mad", "rage", "outraged", "irritated", "upset"],
    "sadness":  ["sad", "depressed", "unhappy", "sorry", "regret",
                 "disappointed", "grief", "miss", "loss", "lonely"],
    "surprise": ["surprised", "shocked", "unexpected", "suddenly",
                 "amazed", "astonished", "wow", "unbelievable", "whoa"],
}

EMOTION_PROMPT = """You are an emotion classifier for a memory system.

Given the following text, return a JSON object with:
- "label": one of ["joy", "fear", "anger", "sadness", "surprise", "neutral"]
- "score": float 0.0-1.0 representing emotional intensity
  (0.0 = completely neutral, 1.0 = extremely intense emotion)
- "reasoning": one sentence explaining your choice

Rules:
- Most factual/informational statements should be "neutral" with score < 0.2
- Only assign non-neutral if there is clear emotional content
- Score reflects intensity, not just presence of emotion

Return ONLY valid JSON, no other text.

Text: "{text}"
"""


class EmotionTagger:
    """
    Tags text with emotional label and intensity before memory storage.

    Usage:
        tagger = EmotionTagger()
        result = tagger.tag("I'm terrified of losing my job")
        # {"label": "fear", "score": 0.85, "reasoning": "..."}

        # Use in store.insert:
        emotion = tagger.tag(text)
        store.insert(text, emotional_label=emotion["label"],
                           emotional_score=emotion["score"])
    """

    def __init__(
        self,
        openai_key: Optional[str] = None,
        mode: str = "llm",   # "llm" or "keyword"
        model: str = "gpt-4o-mini",
    ):
        self.mode  = mode
        self.model = model
        if mode == "llm":
            self._oai = OpenAI(api_key=openai_key or os.getenv("OPENAI_API_KEY"))

    def tag(self, text: str) -> dict:
        """
        Returns dict with keys: label, score, reasoning
        """
        if self.mode == "keyword":
            return self._keyword_tag(text)
        return self._llm_tag(text)

    def tag_and_insert(self, text: str, store, **insert_kwargs) -> str:
        """
        Convenience: tag emotion then insert into store.
        Returns memory UUID.

        Usage:
            mid = tagger.tag_and_insert("I love Python!", store, importance=0.7)
        """
        emotion = self.tag(text)
        return store.insert(
            text,
            emotional_label = emotion["label"],
            emotional_score = emotion["score"],
            **insert_kwargs,
        )

    def _llm_tag(self, text: str) -> dict:
        """Use GPT-4o-mini to classify emotion — accurate but costs tokens."""
        try:
            response = self._oai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": EMOTION_PROMPT.format(text=text[:500])}
                ],
                temperature=0,
                max_tokens=150,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            raw = re.sub(r"```json|```", "", raw).strip()
            result = json.loads(raw)

            # Validate
            if result.get("label") not in EMOTION_LABELS:
                result["label"] = "neutral"
            result["score"] = float(max(0.0, min(1.0, result.get("score", 0.0))))
            return result

        except Exception as e:
            # Fallback to keyword mode on any error
            result = self._keyword_tag(text)
            result["reasoning"] = f"LLM error ({e}), used keyword fallback"
            return result

    def _keyword_tag(self, text: str) -> dict:
        """Fast keyword-based emotion detection — no API call needed."""
        text_lower = text.lower()
        best_label  = "neutral"
        best_count  = 0

        for label, keywords in EMOTION_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > best_count:
                best_count  = count
                best_label  = label

        # Rough score based on keyword density
        score = min(best_count * 0.2, 1.0) if best_count > 0 else 0.0

        return {
            "label":     best_label,
            "score":     score,
            "reasoning": f"Keyword match: {best_count} keywords for '{best_label}'",
        }

    def batch_tag(self, texts: list[str]) -> list[dict]:
        """Tag a list of texts. Useful for bulk imports."""
        return [self.tag(t) for t in texts]
