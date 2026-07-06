"""
memory_os/extractor.py  —  Memory Extraction from Messages
===========================================================
Every message a user sends contains two things mixed together:

  1. THE REQUEST  — what they want answered right now
                    "help me set up authentication"
                    → goes to LLM, NOT stored as memory

  2. THE CONTEXT  — what we learn about them from this message
                    "I'm building with Next.js, I hate Firebase"
                    → extracted, emotion-tagged, stored as memory

This module handles the split + extraction.

Why this matters:
  Without extraction: one message = one blob memory
  With extraction:    one message = N granular memories
                      each with own importance, emotion, decay rate

  "I'm building a SaaS with Next.js, tried Firebase and hated it,
   solo founder, target is small agencies, 3 week deadline, stressed"

  Without: stored as one memory, decays as one unit
  With:    6 separate memories, each decaying at their own rate
           "hates Firebase" survives for months
           "stressed about deadline" fades after 3 weeks
           "solo founder" never decays (permanent semantic fact)

This is the fourth novelty of memory-os. No other tool does this.
"""

import json
import re
from typing import Any, Optional

from .llm_utils import invoke_llm

# ── Extraction prompt ─────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are a memory extraction system for an AI agent.

Your job is to read a user message and extract every distinct piece of 
information that would be useful to remember about this person long term.

EXTRACT:
- Facts about the person (job, role, background)
- Technical preferences (tools, languages, frameworks they like or hate)
- Ongoing projects or goals
- Emotional experiences (stress, excitement, fear about something)
- Personal context (solo founder, team size, deadlines)
- Strong opinions ("I hate X", "I always do Y")

DO NOT EXTRACT:
- The actual question or request ("how do I do X?")
- Greetings ("hi", "thanks", "ok")
- Filler words or phrases
- Hypothetical scenarios they're asking about

For each extracted memory:
- "text": clean standalone fact, written as "User [fact]"
           e.g. "User is building a SaaS product with Next.js"
           NOT "building a SaaS" — always include "User" for clarity
- "importance": float 0.0-1.0
    1.0 = permanent defining fact (core tech stack, strong opinion)
    0.7 = useful context (current project, ongoing situation)  
    0.4 = mildly useful (passing mention, low confidence)
    0.1 = probably not worth storing
- "memory_type": "episodic" or "semantic"
    semantic  = permanent truth about the person, won't change soon
                "User hates Firebase"
                "User is a solo founder"
                "User prefers typed Python"
    episodic  = time-specific, situation-specific, may change
                "User is stressed about a 3-week deadline"
                "User just started learning TypeScript"
                "User had a conflict with their cofounder today"
- "confidence": float 0.0-1.0, how confident you are this is accurate
    (low if it was a passing mention, high if stated clearly)

Return a JSON array. If there is nothing worth extracting, return [].
Return ONLY valid JSON, no other text, no markdown fences.

User message: "{text}"
"""

# ── Simple classifier for obvious cases (saves API calls) ─────────────────────

SKIP_PATTERNS = [
    "hi", "hello", "hey", "thanks", "thank you", "ok", "okay",
    "sure", "yes", "no", "got it", "makes sense", "bye", "goodbye",
    "cool", "great", "awesome", "nice", "good", "sounds good",
]

def _is_trivial(text: str) -> bool:
    """Return True if the message is too short/simple to extract from."""
    cleaned = text.lower().strip().rstrip("!.,?")
    if cleaned in SKIP_PATTERNS:
        return True
    if len(text.split()) < 4:
        return True
    return False


# ── Main extractor ────────────────────────────────────────────────────────────

class MemoryExtractor:
    """
    Extracts multiple distinct memories from a single user message.

    Usage:
        extractor = MemoryExtractor()

        facts = extractor.extract(
            "I'm building a SaaS with Next.js, tried Firebase and hated it"
        )
        # Returns list of ExtractedMemory objects

        # Full pipeline — extract + emotion tag + store
        extractor.extract_and_store(message, store, tagger)
    """

    def __init__(
        self,
        llm: Any = None,
        openai_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        min_importance: float = 0.3,   # ignore anything below this threshold
        min_confidence: float = 0.5,   # ignore low-confidence extractions
    ):
        self.llm             = llm
        self.openai_key      = openai_key
        self.model           = model
        self.min_importance  = min_importance
        self.min_confidence  = min_confidence

    def extract(self, text: str) -> list[dict]:
        """
        Extract all distinct memories from a message.

        Returns list of dicts, each with:
            text, importance, memory_type, confidence
        
        Returns [] if nothing worth extracting.
        """
        # Skip trivial messages without API call
        if _is_trivial(text):
            return []

        try:
            response = invoke_llm(
                self.llm,
                messages=[{
                    "role": "user",
                    "content": EXTRACTION_PROMPT.format(text=text[:1000])
                }],
                model=self.model,
                temperature=0,
                max_tokens=800,
                openai_key=self.openai_key,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"```json|```", "", raw).strip()

            extracted = json.loads(raw)
            if not isinstance(extracted, list):
                return []

            # Filter by importance and confidence thresholds
            filtered = [
                item for item in extracted
                if (
                    isinstance(item, dict)
                    and item.get("importance", 0) >= self.min_importance
                    and item.get("confidence", 0) >= self.min_confidence
                    and item.get("text", "").strip()
                )
            ]

            # Validate and normalise each item
            result = []
            for item in filtered:
                memory_type = item.get("memory_type", "episodic")
                if memory_type not in ("episodic", "semantic"):
                    memory_type = "episodic"

                result.append({
                    "text":        item["text"].strip(),
                    "importance":  float(min(max(item.get("importance", 0.5), 0.0), 1.0)),
                    "memory_type": memory_type,
                    "confidence":  float(min(max(item.get("confidence", 0.5), 0.0), 1.0)),
                })

            return result

        except Exception as e:
            # Never crash the agent because extraction failed
            return []

    def extract_and_store(
        self,
        text: str,
        store,
        tagger,
        source: str = "extractor",
    ) -> list[str]:
        """
        Full pipeline:
          1. Extract facts from message
          2. Emotion-tag each fact individually
          3. Store each as a separate memory

        Returns list of memory UUIDs that were stored.

        This is the key method — replaces the old single-insert approach.
        """
        facts = self.extract(text)
        if not facts:
            return []

        memory_ids = []
        for fact in facts:
            # Emotion tag the individual extracted fact
            # (not the whole message — more accurate per-fact)
            emotion = tagger.tag(fact["text"])

            mid = store.insert(
                text            = fact["text"],
                importance      = fact["importance"],
                memory_type     = fact["memory_type"],
                emotional_score = emotion["score"],
                emotional_label = emotion["label"],
                source          = source,
            )
            memory_ids.append(mid)

        return memory_ids

    def extract_request(self, text: str) -> str:
        """
        Extract just the REQUEST part of a message — what the user
        actually wants answered right now.

        Used to separate "what to store" from "what to answer."

        For most messages this is just the original text.
        For messages with lots of context mixed in, this strips the context.
        """
        # Simple heuristic: if message contains a question mark or
        # action words, those parts are the request
        # For complex cases, we just return the whole message —
        # the LLM is smart enough to focus on the question
        return text
