"""
memory_os/compression.py  —  Memory Consolidation (Sleep Cycle)
================================================================
Clusters episodic memories → summarizes each cluster → stores as semantic.

Novel research contribution: we test whether the LLM summary vector
is semantically equivalent to the centroid of the original cluster.
Results are logged to consolidation_log.jsonl.

Uses litellm — works with any LLM provider.
"""

import os
import json
import time
import math
import logging
from typing import Optional

import numpy as np
try:
    import litellm
except Exception:
    from . import _litellm as litellm
try:
    import instructor
except Exception:
    from . import _instructor as instructor
from .schemas import ConsolidationSummary

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    _HAS_SCHEDULER = True
except ImportError:
    _HAS_SCHEDULER = False

logger = logging.getLogger(__name__)

CONSOLIDATION_LOG = os.getenv("CONSOLIDATION_LOG", "consolidation_log.jsonl")

SUMMARY_PROMPT = """You are a memory consolidation system for an AI agent.

Below are {n} related memories from conversations. Compress them into ONE
semantic insight — what is the key, lasting thing to remember?

Rules:
- One sentence or two at most
- Keep the most important and emotionally significant information
- Write in third person: "The user..." or "The agent knows..."
- Discard trivial details, keep lasting facts and preferences

Memories:
{memories}

Semantic insight:"""


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class MemoryConsolidator:
    """
    Runs the sleep cycle: cluster episodic memories → summarize → store semantic.

    Usage:
        consolidator = MemoryConsolidator(
            store     = store,
            llm_model = "gpt-4o-mini",
            api_key   = "sk-...",
        )
        stats = consolidator.consolidate()
        print(f"Consolidated {stats['clusters_processed']} clusters")

        # Run automatically every N minutes:
        consolidator.start_scheduler(interval_minutes=60)
    """

    def __init__(
        self,
        store,
        llm_model: str = "gpt-4o-mini",
        api_key:   str = "",
        min_cluster_size: int = 3,
        min_importance:   float = 0.4,
        log_path:         str = CONSOLIDATION_LOG,
    ):
        self.store            = store
        self.llm_model        = llm_model
        self.min_cluster_size = min_cluster_size
        self.min_importance   = min_importance
        self.log_path         = log_path

        self._client = instructor.from_litellm(litellm.completion)
        from ._utils import set_litellm_key
        set_litellm_key(llm_model, api_key)

        self._scheduler = None

    def consolidate(self) -> dict:
        """
        Run one consolidation cycle. Returns stats dict.
        """
        try:
            import hdbscan
        except ImportError:
            raise ImportError("pip install hdbscan to use consolidation")

        memories = self.store.get_all(memory_type="episodic", limit=2000)
        if len(memories) < self.min_cluster_size:
            return {"status": "skipped", "reason": "not enough memories", "count": len(memories)}

        # Get vectors and payloads
        vectors  = np.array([m.vector for m in memories])
        payloads = [m.payload for m in memories]
        ids      = [str(m.id) for m in memories]

        # Cluster with HDBSCAN
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, metric="euclidean")
        labels    = clusterer.fit_predict(vectors)

        unique_labels = set(labels) - {-1}  # -1 = noise
        stats = {
            "status":             "ok",
            "total_memories":     len(memories),
            "clusters_found":     len(unique_labels),
            "clusters_processed": 0,
            "memories_deleted":   0,
            "semantic_stored":    0,
        }

        for label in unique_labels:
            mask     = labels == label
            indices  = [i for i, m in enumerate(mask) if m]
            cluster_vectors  = vectors[mask]
            cluster_payloads = [payloads[i] for i in indices]
            cluster_ids      = [ids[i] for i in indices]

            # Skip low-importance clusters
            avg_importance = np.mean([p.get("importance", 0.5) for p in cluster_payloads])
            if avg_importance < self.min_importance:
                continue

            # Summarize cluster
            mem_texts  = [p.get("text", "") for p in cluster_payloads]
            summary_obj = self._summarize(mem_texts)
            if not summary_obj:
                continue

            # Research measurement: summary_vec vs centroid cosine similarity
            centroid     = cluster_vectors.mean(axis=0)
            summary_vec  = self.store.embed(summary_obj.summary)
            cos_sim      = cosine_similarity(np.array(summary_vec), centroid)

            # Store as semantic memory
            dominant_emotion = max(
                cluster_payloads,
                key=lambda p: p.get("emotional_score", 0.0),
            )
            mid = self.store.insert(
                text            = summary_obj.summary,
                importance      = float(avg_importance),
                memory_type     = "semantic",
                emotional_score = summary_obj.emotional_score,
                emotional_label = summary_obj.key_emotion,
                source          = "consolidator",
                extra_payload   = {
                    "source_count":       len(cluster_ids),
                    "centroid_cos_sim":   round(cos_sim, 4),
                    "cluster_label":      int(label),
                },
            )

            # Delete source episodic memories
            for mid_del in cluster_ids:
                try:
                    self.store.delete(mid_del)
                except Exception:
                    pass

            # Log research data
            self._log({
                "timestamp":         time.time(),
                "cluster_size":      len(cluster_ids),
                "centroid_cos_sim":  cos_sim,
                "avg_importance":    float(avg_importance),
                "key_emotion":       summary_obj.key_emotion,
                "confidence":        summary_obj.confidence,
                "summary":           summary_obj.summary,
            })

            stats["clusters_processed"] += 1
            stats["memories_deleted"]   += len(cluster_ids)
            stats["semantic_stored"]    += 1

        return stats

    def _summarize(self, texts: list[str]) -> Optional[ConsolidationSummary]:
        try:
            return self._client.chat.completions.create(
                model          = self.llm_model,
                response_model = ConsolidationSummary,
                max_retries    = 2,
                messages       = [{
                    "role":    "user",
                    "content": SUMMARY_PROMPT.format(
                        n=len(texts),
                        memories="\n".join(f"- {t}" for t in texts),
                    ),
                }],
                temperature = 0,
                max_tokens  = 200,
            )
        except Exception as e:
            logger.warning(f"Consolidation summarize failed: {e}")
            return None

    def _log(self, data: dict) -> None:
        try:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(data) + "\n")
        except Exception:
            pass

    def start_scheduler(self, interval_minutes: int = 60) -> None:
        """Run consolidation in background every N minutes."""
        if not _HAS_SCHEDULER:
            raise ImportError("pip install apscheduler to use scheduler")
        self._scheduler = BackgroundScheduler()
        self._scheduler.add_job(self.consolidate, "interval", minutes=interval_minutes)
        self._scheduler.start()
        logger.info(f"Consolidation scheduler started — runs every {interval_minutes} min")

    def stop_scheduler(self) -> None:
        if self._scheduler:
            self._scheduler.shutdown()
