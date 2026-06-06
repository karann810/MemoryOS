"""
memory_os/compression.py  —  Week 3: Memory Consolidation (Sleep Cycle)
========================================================================
This implements the most novel research contribution of memory-os.

The question (Gap 2 from our research framing):
  When you cluster episodic memories and summarize them into a semantic
  memory — is the summary vector semantically equivalent to the centroid
  of the cluster?

  This has NEVER been tested or published. We test it here and log results.

What this module does:
  1. Runs as a background job (APScheduler, every N conversations or nightly)
  2. Fetches all episodic memories from Qdrant
  3. Clusters them with HDBSCAN (density-based, handles noise)
  4. For each cluster:
     a. Asks LLM to summarize the cluster into one semantic insight
     b. Embeds the summary → summary_vector
     c. Computes centroid of the cluster's raw vectors
     d. Logs cosine similarity(summary_vector, centroid)  ← the research question
     e. Stores summary as a new "semantic" memory
     f. Deletes the original episodic memories
  5. Saves consolidation stats to docs/consolidation_log.jsonl for analysis

The logged data answers Gap 2 and is publishable.
"""

import os
import json
import time
import math
import logging
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from apscheduler.schedulers.background import BackgroundScheduler

load_dotenv()
logger = logging.getLogger(__name__)

CONSOLIDATION_LOG = os.getenv("CONSOLIDATION_LOG", "docs/consolidation_log.jsonl")

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
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class MemoryConsolidator:
    """
    Runs the sleep cycle: cluster episodic memories → summarize → store semantic.

    Usage:
        consolidator = MemoryConsolidator(store)

        # Run once manually
        stats = consolidator.run()

        # Schedule nightly
        consolidator.schedule(hour=3, minute=0)
    """

    def __init__(
        self,
        store,                           # MemoryStore instance
        openai_key: Optional[str] = None,
        min_cluster_size: int = 3,       # HDBSCAN: min memories to form a cluster
        min_memories_to_run: int = 10,   # don't consolidate until we have enough
        log_path: str = CONSOLIDATION_LOG,
    ):
        self.store              = store
        self.min_cluster_size   = min_cluster_size
        self.min_memories_to_run = min_memories_to_run
        self.log_path           = log_path
        self._oai               = OpenAI(api_key=openai_key or os.getenv("OPENAI_API_KEY"))
        self._scheduler         = None

        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def run(self) -> dict:
        """
        Run one consolidation cycle.
        Returns stats dict: clusters_found, memories_compressed, avg_centroid_similarity
        """
        logger.info("[Consolidator] Starting sleep cycle...")

        # 1. Fetch all episodic memories
        episodic = self.store.get_all(memory_type="episodic")
        if len(episodic) < self.min_memories_to_run:
            logger.info(f"[Consolidator] Only {len(episodic)} episodic memories — skipping.")
            return {"status": "skipped", "reason": "not enough memories"}

        logger.info(f"[Consolidator] Clustering {len(episodic)} episodic memories...")

        # 2. Extract vectors
        vectors = np.array([m.vector for m in episodic])
        ids     = [m.id for m in episodic]
        texts   = [m.payload.get("text", "") for m in episodic]

        # 3. Cluster with HDBSCAN
        try:
            import hdbscan
        except ImportError:
            raise ImportError("pip install hdbscan")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric="euclidean",
            prediction_data=True,
        )
        labels = clusterer.fit_predict(vectors)

        unique_labels = set(labels) - {-1}  # -1 = noise
        logger.info(f"[Consolidator] Found {len(unique_labels)} clusters, "
                    f"{(labels == -1).sum()} noise points")

        stats = {
            "timestamp":             time.time(),
            "episodic_count":        len(episodic),
            "clusters_found":        len(unique_labels),
            "memories_compressed":   0,
            "centroid_similarities": [],
        }

        # 4. Process each cluster
        for cluster_id in unique_labels:
            mask        = labels == cluster_id
            cluster_ids = [ids[i]   for i, m in enumerate(mask) if m]
            cluster_txt = [texts[i] for i, m in enumerate(mask) if m]
            cluster_vecs = vectors[mask]

            # a. Summarize with LLM
            summary = self._summarize(cluster_txt)
            if not summary:
                continue

            # b. Embed summary
            summary_vector = self._embed(summary)

            # c. Compute centroid of cluster
            centroid = cluster_vecs.mean(axis=0)

            # d. THE RESEARCH MEASUREMENT: similarity(summary, centroid)
            sim = cosine_similarity(np.array(summary_vector), centroid)
            stats["centroid_similarities"].append(sim)
            logger.info(f"[Consolidator] Cluster {cluster_id}: "
                        f"{len(cluster_ids)} memories → sim={sim:.4f}")

            # Inherit emotional weight from strongest memory in cluster
            payloads = [m.payload for m in episodic if m.id in cluster_ids]
            max_emotional = max(
                (p.get("emotional_score", 0.0) for p in payloads), default=0.0
            )
            dominant_emotion = max(
                payloads,
                key=lambda p: p.get("emotional_score", 0.0),
                default={}
            ).get("emotional_label", "neutral")

            # e. Store as semantic memory
            self.store.insert(
                text           = summary,
                importance     = min(0.5 + (len(cluster_ids) * 0.05), 1.0),
                memory_type    = "semantic",
                emotional_score = max_emotional,
                emotional_label = dominant_emotion,
                source         = "consolidator",
                extra_payload  = {
                    "source_count":         len(cluster_ids),
                    "centroid_similarity":  sim,
                    "cluster_id":           int(cluster_id),
                },
            )

            # f. Delete originals
            for mid in cluster_ids:
                self.store.delete(mid)

            stats["memories_compressed"] += len(cluster_ids)

        # Compute avg similarity (the key research metric)
        if stats["centroid_similarities"]:
            stats["avg_centroid_similarity"] = sum(stats["centroid_similarities"]) / len(
                stats["centroid_similarities"]
            )
        else:
            stats["avg_centroid_similarity"] = None

        # 5. Log for research analysis
        self._log(stats)
        logger.info(f"[Consolidator] Done. Compressed {stats['memories_compressed']} memories. "
                    f"Avg centroid sim: {stats.get('avg_centroid_similarity', 'N/A'):.4f}")
        return stats

    def schedule(self, hour: int = 3, minute: int = 0) -> None:
        """Schedule the sleep cycle to run nightly at hour:minute."""
        self._scheduler = BackgroundScheduler()
        self._scheduler.add_job(self.run, "cron", hour=hour, minute=minute)
        self._scheduler.start()
        logger.info(f"[Consolidator] Scheduled nightly at {hour:02d}:{minute:02d}")

    def stop_schedule(self) -> None:
        if self._scheduler:
            self._scheduler.shutdown()

    def _summarize(self, texts: list[str]) -> Optional[str]:
        """Ask LLM to compress a cluster of texts into one semantic insight."""
        memories_str = "\n".join(f"- {t}" for t in texts)
        try:
            response = self._oai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": SUMMARY_PROMPT.format(
                        n=len(texts), memories=memories_str
                    )
                }],
                temperature=0.3,
                max_tokens=150,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"[Consolidator] Summarization failed: {e}")
            return None

    def _embed(self, text: str) -> list[float]:
        response = self._oai.embeddings.create(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            input=text,
        )
        return response.data[0].embedding

    def _log(self, stats: dict) -> None:
        """Append consolidation stats to JSONL file for research analysis."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(stats) + "\n")
