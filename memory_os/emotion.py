"""
memory_os/emotion.py  —  Emotional Tagger (Instructor + LiteLLM)
=================================================================
Tags every memory with emotional label + intensity before storage.

Two modes:
  "llm" (default) — accurate, uses Instructor
  "off"           — skips tagging, everything = neutral (faster/cheaper)
"""

try:
    import litellm
except Exception:
    from . import _litellm as litellm
try:
    import instructor
except Exception:
    from . import _instructor as instructor
from ._utils import set_litellm_key
from .schemas import EmotionResult

EMOTION_PROMPT = """You are an emotion classifier for a memory system.

Classify the emotion in the following text.

Rules:
- Most factual/informational statements = neutral with low score
- Understand negation: "not happy" is NOT joy
- Only assign non-neutral if there is CLEAR emotional content

Text: "{text}"
"""

NEUTRAL_RESULT = EmotionResult(
    label="neutral",
    score=0.0,
    reasoning="Emotion tagging disabled (mode=off)",
)


class EmotionTagger:
    """
    Tags text with emotional label and intensity before memory storage.

    Usage:
        tagger = EmotionTagger(model="gpt-4o-mini", api_key="sk-...")
        result = tagger.tag("I'm terrified of losing my job")
        print(result.label, result.score)   # fear, 0.85

        # Disable for testing / cost savings
        tagger = EmotionTagger(model="gpt-4o-mini", api_key="sk-...", mode="off")
    """

    def __init__(
        self,
        model:   str = "gpt-4o-mini",
        api_key: str = "",
        mode:    str = "llm",
    ):
        if mode not in ("llm", "off", "keyword"):
            raise ValueError(f"Invalid mode '{mode}'. Use 'llm', 'off' or 'keyword'.")
        self.mode  = mode
        self.model = model
        if mode == "llm":
            self._client = instructor.from_litellm(litellm.completion)
            set_litellm_key(model, api_key)

        # Keyword mode: simple heuristic-based tagging (fast, deterministic)
        if mode == "keyword":
            self.mode = "keyword"

    def tag(self, text: str) -> EmotionResult:
        """Tag text with emotion. Never crashes — returns neutral on failure."""
        if self.mode == "off":
            return NEUTRAL_RESULT
        if self.mode == "keyword":
            t = text.lower()
            if any(w in t for w in ("happy", "joy", "excited", "love")):
                res = EmotionResult(label="joy", score=0.6, reasoning="Keyword match")
                return res.model_dump() if self.mode == "keyword" else res
            if any(w in t for w in ("scared", "afraid", "fear", "terrified")):
                res = EmotionResult(label="fear", score=0.7, reasoning="Keyword match")
                return res.model_dump() if self.mode == "keyword" else res
            if any(w in t for w in ("hate", "angry", "furious", "annoyed")):
                res = EmotionResult(label="anger", score=0.7, reasoning="Keyword match")
                return res.model_dump() if self.mode == "keyword" else res
            if any(w in t for w in ("sad", "depressed", "unhappy", "sorrow")):
                res = EmotionResult(label="sadness", score=0.6, reasoning="Keyword match")
                return res.model_dump() if self.mode == "keyword" else res
            if "surpris" in t or "wow" in t:
                res = EmotionResult(label="surprise", score=0.5, reasoning="Keyword match")
                return res.model_dump() if self.mode == "keyword" else res
            res = EmotionResult(label="neutral", score=0.0, reasoning="No emotion keywords found")
            return res.model_dump() if self.mode == "keyword" else res
        try:
            return self._client.chat.completions.create(
                model          = self.model,
                response_model = EmotionResult,
                max_retries    = 2,
                messages       = [{
                    "role":    "user",
                    "content": EMOTION_PROMPT.format(text=text[:500]),
                }],
                temperature = 0,
                max_tokens  = 100,
            )
        except Exception as e:
            return EmotionResult(
                label     = "neutral",
                score     = 0.0,
                reasoning = f"Tagging failed: {str(e)[:80]}",
            )

    def batch_tag(self, texts: list[str]) -> list[EmotionResult]:
        return [self.tag(t) for t in texts]
