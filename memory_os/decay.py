"""
memory_os/decay.py  —  Week 2: Ebbinghaus Forgetting Curve Re-ranker
=====================================================================
This is the core research novelty of memory-os.

Every existing tool (Mem0, Zep, LangChain) uses one of:
  - Raw cosine similarity only
  - Simple linear time decay:  score * (1 / days_elapsed)

We use the actual Ebbinghaus forgetting curve from 1885 psychology research:

    R(t) = e^(-t / S)

Where:
  R = retention strength (0-1)
  t = time elapsed since last access (in days)
  S = memory stability — increases with each successful retrieval
      (this models the "spacing effect": reviewing a memory makes it stronger)

Combined formula:
    M = similarity × R(t,S) × importance × emotional_weight

The emotional_weight multiplier is the second novelty:
  Emotionally charged memories (fear, joy, anger) are retained
  1.5-2x longer than neutral ones. This is well-documented in
  psychology (McGaugh 2000, "Memory and Emotion").

References:
  - Ebbinghaus, H. (1885). Über das Gedächtnis.
  - Wozniak & Gorzelanczyk (1994). SuperMemo spacing algorithm.
  - McGaugh, J.L. (2000). Memory — a century of consolidation. Science.
"""

import math
import time
from dataclasses import dataclass
from typing import Optional


# Emotional label → how much it boosts memory stability
# Based on McGaugh (2000): amygdala activation during emotional events
# strengthens hippocampal consolidation
EMOTION_STABILITY_BOOST = {
    "fear":     2.0,   # highest — survival mechanism
    "anger":    1.8,
    "joy":      1.6,
    "surprise": 1.5,
    "sadness":  1.4,
    "neutral":  1.0,   # baseline
}

# Base stability (days) for a memory never accessed after encoding
# Below this threshold retrieval probability < 10%
BASE_STABILITY_DAYS = 1.0


@dataclass
class RankedMemory:
    """A memory with its final composite score and score breakdown."""
    id: str
    text: str
    payload: dict
    similarity: float         # raw cosine similarity from Qdrant
    retention: float          # Ebbinghaus R(t, S)
    emotional_weight: float   # emotion boost factor
    importance: float         # stored importance
    final_score: float        # M = similarity × retention × emotional_weight × importance

    def __repr__(self):
        return (
            f"RankedMemory(score={self.final_score:.3f}, "
            f"sim={self.similarity:.3f}, ret={self.retention:.3f}, "
            f"emo={self.emotional_weight:.2f}, "
            f"text='{self.text[:60]}...')"
        )


def ebbinghaus_retention(
    created_at: float,
    last_accessed: float,
    access_history: list[float],
    emotional_label: str = "neutral",
    emotional_score: float = 0.0,
) -> tuple[float, float]:
    """
    Compute memory retention R and stability S using the Ebbinghaus model.

    Stability S grows with each spaced retrieval (spacing effect):
      S_n = S_{n-1} × spacing_factor  if gap >= S_{n-1}
      S_n = S_{n-1}                   if reviewed too soon (no benefit)

    Returns (retention, stability) both floats.
    """
    now = time.time()

    # Emotional boost on base stability
    emotion_boost = EMOTION_STABILITY_BOOST.get(emotional_label, 1.0)
    # Additional boost from emotional intensity (0-1 score)
    intensity_boost = 1.0 + (emotional_score * 0.5)
    S = BASE_STABILITY_DAYS * emotion_boost * intensity_boost

    # Simulate stability growth through the access history (spacing effect)
    # Sort accesses chronologically
    history = sorted(access_history or [])
    prev_time = created_at

    for access_time in history:
        gap_days = (access_time - prev_time) / 86400.0
        if gap_days >= S:
            # Spaced enough — memory consolidates, stability grows
            S = S * (1.5 + min(gap_days / S, 3.0))
        # else: reviewed too soon, no stability gain (forgetting curve resets)
        prev_time = access_time

    # Time since last access
    last = last_accessed if last_accessed else created_at
    t_days = max((now - last) / 86400.0, 0.0)

    # Ebbinghaus: R = e^(-t/S)
    retention = math.exp(-t_days / max(S, 0.001))

    return retention, S


def compute_emotional_weight(
    emotional_label: str,
    emotional_score: float,
) -> float:
    """
    Emotional weight for the final score.
    Ranges from 1.0 (neutral) to ~2.0 (intense fear).
    """
    base = EMOTION_STABILITY_BOOST.get(emotional_label, 1.0)
    intensity = 1.0 + (emotional_score * 0.5)
    # Normalize so neutral memory has weight 1.0
    return base * intensity


class DecayReranker:
    """
    Re-ranks raw Qdrant results using the Ebbinghaus forgetting curve
    combined with emotional weighting and importance.

    Usage:
        reranker = DecayReranker()
        raw_results = store.retrieve("What does user prefer?", top_k=50)
        top5 = reranker.rerank(raw_results, top_n=5)
        for mem in top5:
            print(mem.final_score, mem.text)
    """

    def __init__(
        self,
        similarity_weight: float = 1.0,
        retention_weight: float = 1.0,
        importance_weight: float = 1.0,
        emotion_weight: float = 1.0,
    ):
        """
        Weights let you tune how much each factor contributes.
        Default is equal weighting — good starting point for research.
        """
        self.w_sim  = similarity_weight
        self.w_ret  = retention_weight
        self.w_imp  = importance_weight
        self.w_emo  = emotion_weight

    def score(self, result) -> RankedMemory:
        """Compute full psychological score for a single Qdrant ScoredPoint."""
        p = result.payload
        similarity = result.score

        retention, stability = ebbinghaus_retention(
            created_at      = p.get("created_at",    time.time()),
            last_accessed   = p.get("last_accessed", time.time()),
            access_history  = p.get("access_history", []),
            emotional_label = p.get("emotional_label", "neutral"),
            emotional_score = p.get("emotional_score", 0.0),
        )

        emotional_weight = compute_emotional_weight(
            emotional_label = p.get("emotional_label", "neutral"),
            emotional_score = p.get("emotional_score", 0.0),
        )

        importance = p.get("importance", 0.5)

        # Composite score — multiplicative so a zero in any dimension kills it
        final_score = (
            (similarity      ** self.w_sim) *
            (retention       ** self.w_ret) *
            (importance      ** self.w_imp) *
            (emotional_weight ** self.w_emo)
        )

        return RankedMemory(
            id               = str(result.id),
            text             = p.get("text", ""),
            payload          = p,
            similarity       = similarity,
            retention        = retention,
            emotional_weight = emotional_weight,
            importance       = importance,
            final_score      = final_score,
        )

    def rerank(self, raw_results: list, top_n: int = 5) -> list[RankedMemory]:
        """
        Score and sort raw Qdrant results.
        Returns top_n RankedMemory objects, best first.
        """
        scored = [self.score(r) for r in raw_results]
        scored.sort(key=lambda m: m.final_score, reverse=True)
        return scored[:top_n]

    def rerank_with_explanation(self, raw_results: list, top_n: int = 5):
        """
        Same as rerank but also returns the full scored list
        so you can see what got promoted vs demoted.
        Useful for the Week 8 benchmark and blog post.
        """
        scored = [self.score(r) for r in raw_results]
        scored.sort(key=lambda m: m.final_score, reverse=True)

        return {
            "top":      scored[:top_n],
            "promoted": [m for m in scored[:top_n] if scored.index(m) < raw_results.index(
                next(r for r in raw_results if str(r.id) == m.id), 999
            )],
            "all":      scored,
        }
