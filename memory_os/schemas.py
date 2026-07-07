"""
memory_os/schemas.py  —  All Pydantic schemas for LLM outputs
=============================================================
Every LLM call in memory-os returns one of these schemas.
No json.loads(). No regex. No crashes from malformed output.

How it works:
  Normal:     LLM returns text → you parse it → crashes possible
  With this:  LLM is forced into exact schema via Instructor
              You get a Pydantic object back, always valid, always typed

Usage (internal — you don't use these directly):
  from memory_os.schemas import EmotionResult, ExtractionResult
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal


# ── Emotion tagging ───────────────────────────────────────────────────────────

class EmotionResult(BaseModel):
    """
    Output of EmotionTagger.tag()

    label:     one of 6 emotions or neutral
    score:     0.0 = completely neutral, 1.0 = extremely intense
    reasoning: one sentence explanation
    """
    label: Literal["joy", "fear", "anger", "sadness", "surprise", "neutral"]
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(max_length=200)

    @field_validator("score")
    @classmethod
    def round_score(cls, v: float) -> float:
        return round(v, 3)


# ── Memory extraction ─────────────────────────────────────────────────────────

class ExtractedFact(BaseModel):
    """
    A single fact extracted from a user message.

    text:        clean standalone fact, written as "User [fact]"
    importance:  0.0 = not worth storing, 1.0 = permanent defining fact
    memory_type: episodic = time-specific event
                 semantic = permanent truth about the person
    confidence:  how confident the LLM is about this extraction
    emotion:     emotion tag for this specific fact (combined call)
    """
    text: str = Field(min_length=5, max_length=300)
    importance: float = Field(ge=0.0, le=1.0)
    memory_type: Literal["episodic", "semantic"]
    confidence: float = Field(ge=0.0, le=1.0)
    emotion: EmotionResult

    @field_validator("text")
    @classmethod
    def clean_text(cls, v: str) -> str:
        v = v.strip()
        # Ensure it starts with "User" for clarity
        if not v.lower().startswith("user"):
            v = f"User {v[0].lower()}{v[1:]}"
        return v

    @field_validator("importance", "confidence")
    @classmethod
    def round_floats(cls, v: float) -> float:
        return round(v, 3)


class ExtractionResult(BaseModel):
    """
    Full output of MemoryExtractor.extract()
    Wraps a list of extracted facts.
    """
    facts: list[ExtractedFact] = Field(default_factory=list)

    @property
    def has_facts(self) -> bool:
        return len(self.facts) > 0

    @property
    def semantic_facts(self) -> list[ExtractedFact]:
        return [f for f in self.facts if f.memory_type == "semantic"]

    @property
    def episodic_facts(self) -> list[ExtractedFact]:
        return [f for f in self.facts if f.memory_type == "episodic"]


# ── Memory consolidation (sleep cycle) ───────────────────────────────────────

class ConsolidationSummary(BaseModel):
    """
    Output of the LLM summarisation step in compression.py

    summary:          the compressed semantic insight
    key_emotion:      dominant emotion across the cluster
    emotional_score:  intensity of that emotion
    confidence:       how confident the LLM is in the summary
    """
    summary: str = Field(min_length=10, max_length=300)
    key_emotion: Literal["joy", "fear", "anger", "sadness", "surprise", "neutral"]
    emotional_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("summary")
    @classmethod
    def clean_summary(cls, v: str) -> str:
        return v.strip()
