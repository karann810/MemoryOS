#!/usr/bin/env python3
"""
benchmark/run_benchmark.py  —  Week 8: Does memory-os beat vanilla RAG?
========================================================================
Runs a controlled experiment comparing:
  Condition A: Vanilla RAG (cosine similarity only)
  Condition B: memory-os (Ebbinghaus + emotion + importance)

Tasks (3-5 multi-turn scenarios):
  1. Preference recall     — does the agent remember user preferences?
  2. Emotional event recall — are emotional memories retained longer?
  3. Fact decay            — do old, unimportant facts correctly fade?
  4. Spaced repetition     — does re-accessing a memory strengthen it?
  5. Contradiction handling — does new info appropriately update old?

Each task has a ground-truth answer. We measure:
  - Recall@5: is the correct memory in the top 5?
  - MRR: mean reciprocal rank of the correct memory
  - Score delta: memory-os score vs vanilla RAG score

Output: benchmark/results.json + benchmark/results.md (for blog post)
"""

import os
import sys
import json
import time
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from memory_os.store import MemoryStore
from memory_os.decay import DecayReranker
from memory_os.emotion import EmotionTagger
from dotenv import load_dotenv

load_dotenv()

# ── Benchmark dataset ─────────────────────────────────────────────────────────

BENCHMARK_TASKS = [
    {
        "id":          "pref_recall",
        "description": "Preference recall",
        "setup_memories": [
            {"text": "User said they hate verbose explanations.",
             "importance": 0.9, "emotional_label": "anger",
             "emotional_score": 0.7, "days_old": 5},
            {"text": "User prefers Python over JavaScript.",
             "importance": 0.8, "emotional_label": "neutral",
             "emotional_score": 0.1, "days_old": 10},
            {"text": "User mentioned they enjoy hiking on weekends.",
             "importance": 0.5, "emotional_label": "joy",
             "emotional_score": 0.5, "days_old": 20},
            # Noise memories
            {"text": "User asked about the weather yesterday.",
             "importance": 0.2, "emotional_label": "neutral",
             "emotional_score": 0.0, "days_old": 1},
            {"text": "User said hi.",
             "importance": 0.1, "emotional_label": "neutral",
             "emotional_score": 0.0, "days_old": 2},
        ],
        "query":          "What programming language does the user prefer?",
        "correct_memory": "User prefers Python over JavaScript.",
    },
    {
        "id":          "emotional_retention",
        "description": "Emotional memory retention",
        "setup_memories": [
            {"text": "User was terrified when they lost all their project files.",
             "importance": 0.6, "emotional_label": "fear",
             "emotional_score": 0.95, "days_old": 30},
            {"text": "User mentioned they had lunch.",
             "importance": 0.2, "emotional_label": "neutral",
             "emotional_score": 0.0, "days_old": 2},
            {"text": "User asked what time it is.",
             "importance": 0.1, "emotional_label": "neutral",
             "emotional_score": 0.0, "days_old": 5},
        ],
        "query":          "Has the user had any scary experiences?",
        "correct_memory": "User was terrified when they lost all their project files.",
    },
    {
        "id":          "fact_decay",
        "description": "Old unimportant facts should fade",
        "setup_memories": [
            {"text": "User's meeting was rescheduled to 3pm on Monday.",
             "importance": 0.2, "emotional_label": "neutral",
             "emotional_score": 0.0, "days_old": 60},   # very old, low importance
            {"text": "User always starts coding sessions with a coffee.",
             "importance": 0.7, "emotional_label": "joy",
             "emotional_score": 0.4, "days_old": 3},    # recent, higher importance
        ],
        "query":         "What does the user do when starting to code?",
        "correct_memory": "User always starts coding sessions with a coffee.",
    },
    {
        "id":          "spaced_repetition",
        "description": "Repeatedly accessed memories stay strong",
        "setup_memories": [
            {"text": "User is building a memory OS for AI agents.",
             "importance": 0.8, "emotional_label": "joy",
             "emotional_score": 0.6, "days_old": 20,
             "access_history_days": [19, 15, 10, 5, 1]},  # accessed regularly
            {"text": "User once mentioned they like jazz music.",
             "importance": 0.4, "emotional_label": "neutral",
             "emotional_score": 0.1, "days_old": 20,
             "access_history_days": []},  # never accessed since
        ],
        "query":          "What project is the user working on?",
        "correct_memory": "User is building a memory OS for AI agents.",
    },
]


# ── Scoring helpers ───────────────────────────────────────────────────────────

def recall_at_k(results: list, correct_text: str, k: int = 5) -> float:
    texts = [r.get("text", "") if isinstance(r, dict) else r.payload.get("text", "")
             for r in results[:k]]
    return 1.0 if any(correct_text[:50] in t for t in texts) else 0.0


def reciprocal_rank(results: list, correct_text: str) -> float:
    for i, r in enumerate(results):
        text = r.get("text", "") if isinstance(r, dict) else r.payload.get("text", "")
        if correct_text[:50] in text:
            return 1.0 / (i + 1)
    return 0.0


# ── Main benchmark ────────────────────────────────────────────────────────────

def run_benchmark():
    store    = MemoryStore(collection="benchmark_memories")
    reranker = DecayReranker()
    now      = time.time()

    results_summary = {
        "timestamp":    now,
        "tasks":        [],
        "vanilla_mrr":  0.0,
        "memoryos_mrr": 0.0,
        "vanilla_recall@5":   0.0,
        "memoryos_recall@5":  0.0,
    }

    print("\n" + "=" * 60)
    print("  memory-os Benchmark")
    print("  Vanilla RAG vs Ebbinghaus + Emotion + Importance")
    print("=" * 60)

    for task in BENCHMARK_TASKS:
        print(f"\n📋 Task: {task['description']}")
        store.wipe()

        # Insert memories at correct ages
        for mem in task["setup_memories"]:
            days_old = mem.get("days_old", 0)
            created  = now - (days_old * 86400)

            # Build access history
            access_history = []
            for d in mem.get("access_history_days", []):
                access_history.append(now - (d * 86400))

            mid = store.insert(
                text            = mem["text"],
                importance      = mem.get("importance", 0.5),
                emotional_label = mem.get("emotional_label", "neutral"),
                emotional_score = mem.get("emotional_score", 0.0),
                source          = "benchmark",
            )
            # Backdate the timestamps
            store.update_payload(mid, {
                "created_at":    created,
                "last_accessed": created if not access_history else max(access_history),
                "access_history": access_history,
            })

        # Retrieve
        raw_results = store.retrieve(task["query"], top_k=50, update_access=False)

        # Condition A: Vanilla RAG (cosine similarity order)
        vanilla_rr  = reciprocal_rank(raw_results, task["correct_memory"])
        vanilla_r5  = recall_at_k(raw_results, task["correct_memory"], k=5)

        # Condition B: memory-os (Ebbinghaus + emotion + importance)
        ranked      = reranker.rerank(raw_results, top_n=len(raw_results))
        memoryos_rr = reciprocal_rank(ranked, task["correct_memory"])
        memoryos_r5 = recall_at_k(ranked, task["correct_memory"], k=5)

        task_result = {
            "id":              task["id"],
            "description":     task["description"],
            "query":           task["query"],
            "correct_memory":  task["correct_memory"],
            "vanilla_rr":      vanilla_rr,
            "memoryos_rr":     memoryos_rr,
            "vanilla_recall5": vanilla_r5,
            "memoryos_recall5": memoryos_r5,
            "improved":        memoryos_rr > vanilla_rr,
        }
        results_summary["tasks"].append(task_result)

        verdict = "✅ IMPROVED" if memoryos_rr > vanilla_rr else (
                  "➡️  TIED"    if memoryos_rr == vanilla_rr else
                  "❌ WORSE")
        print(f"  Vanilla  RR={vanilla_rr:.3f}  Recall@5={vanilla_r5:.1f}")
        print(f"  memory-os RR={memoryos_rr:.3f}  Recall@5={memoryos_r5:.1f}  {verdict}")

    # Aggregate
    tasks = results_summary["tasks"]
    results_summary["vanilla_mrr"]      = sum(t["vanilla_rr"]      for t in tasks) / len(tasks)
    results_summary["memoryos_mrr"]     = sum(t["memoryos_rr"]     for t in tasks) / len(tasks)
    results_summary["vanilla_recall@5"] = sum(t["vanilla_recall5"] for t in tasks) / len(tasks)
    results_summary["memoryos_recall@5"]= sum(t["memoryos_recall5"] for t in tasks) / len(tasks)

    print("\n" + "=" * 60)
    print(f"  Vanilla RAG   MRR={results_summary['vanilla_mrr']:.3f}  "
          f"Recall@5={results_summary['vanilla_recall@5']:.2f}")
    print(f"  memory-os     MRR={results_summary['memoryos_mrr']:.3f}  "
          f"Recall@5={results_summary['memoryos_recall@5']:.2f}")
    delta = results_summary["memoryos_mrr"] - results_summary["vanilla_mrr"]
    print(f"  Delta MRR: {delta:+.3f} ({'better' if delta > 0 else 'worse'})")
    print("=" * 60)

    # Save results
    os.makedirs("benchmark", exist_ok=True)
    with open("benchmark/results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    _write_markdown_report(results_summary)
    print("\n📄 Results saved to benchmark/results.json and benchmark/results.md")

    store.wipe()
    return results_summary


def _write_markdown_report(results: dict) -> None:
    lines = [
        "# memory-os Benchmark Results",
        "",
        "## Does Ebbinghaus + Emotional Weighting beat Vanilla RAG?",
        "",
        "| Metric | Vanilla RAG | memory-os | Delta |",
        "|--------|------------|-----------|-------|",
        f"| MRR | {results['vanilla_mrr']:.3f} | {results['memoryos_mrr']:.3f} | "
        f"{results['memoryos_mrr'] - results['vanilla_mrr']:+.3f} |",
        f"| Recall@5 | {results['vanilla_recall@5']:.2f} | {results['memoryos_recall@5']:.2f} | "
        f"{results['memoryos_recall@5'] - results['vanilla_recall@5']:+.2f} |",
        "",
        "## Per-task breakdown",
        "",
        "| Task | Vanilla MRR | memory-os MRR | Result |",
        "|------|------------|---------------|--------|",
    ]
    for t in results["tasks"]:
        verdict = "✅ Improved" if t["improved"] else "➡️ Tied / Worse"
        lines.append(
            f"| {t['description']} | {t['vanilla_rr']:.3f} | {t['memoryos_rr']:.3f} | {verdict} |"
        )
    lines += [
        "",
        "## Methodology",
        "",
        "- Each task inserts memories at controlled ages and importance levels",
        "- Condition A: raw cosine similarity ranking from Qdrant",
        "- Condition B: re-ranked with `score = similarity × R(t,S) × importance × emotional_weight`",
        "- R(t,S) = Ebbinghaus retention: `e^(-t/S)` where S grows with spaced retrieval",
        "- Emotional weight from McGaugh (2000): fear=2.0x, anger=1.8x, joy=1.6x, neutral=1.0x",
        "",
        "## References",
        "",
        "- Ebbinghaus, H. (1885). Über das Gedächtnis.",
        "- McGaugh, J.L. (2000). Memory — a century of consolidation. Science, 287(5451).",
        "- Wozniak & Gorzelanczyk (1994). Optimization of repetition spacing in the expansion-rehearsal system.",
    ]
    with open("benchmark/results.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_benchmark()
