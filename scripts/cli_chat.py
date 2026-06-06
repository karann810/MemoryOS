#!/usr/bin/env python3
"""
scripts/cli_chat.py  —  Week 2 Demo: CLI chat with decay re-ranker
===================================================================
Run: python scripts/cli_chat.py

Have 10+ conversations, then watch what gets remembered vs forgotten.
This is the demo for the LinkedIn post: "I ran 10 conversations through
my AI agent. Here's what it forgot and why."
"""

import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from memory_os.store import MemoryStore
from memory_os.decay import DecayReranker
from memory_os.emotion import EmotionTagger
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

SYSTEM = """You are a helpful AI assistant with memory.
Use the provided memories to give personalized responses.
Be concise."""

def build_prompt(query: str, memories) -> str:
    if not memories:
        return query
    mem_lines = []
    for m in memories:
        ret_pct = int(m.retention * 100)
        emo     = m.payload.get("emotional_label", "neutral")
        mem_lines.append(
            f"[retention={ret_pct}%, emotion={emo}] {m.text}"
        )
    context = "\n".join(mem_lines)
    return f"Memories:\n{context}\n\nUser: {query}"


def print_memory_debug(ranked_memories):
    print("\n  📊 Memory debug (what survived decay):")
    for i, m in enumerate(ranked_memories):
        ret_pct = int(m.retention * 100)
        emo     = m.payload.get("emotional_label", "neutral")
        age_days = (time.time() - m.payload.get("created_at", time.time())) / 86400
        print(f"  {i+1}. [{ret_pct}% retained | {age_days:.1f}d old | {emo}] "
              f"{m.text[:70]}...")
    print()


def main():
    store    = MemoryStore()
    reranker = DecayReranker()
    tagger   = EmotionTagger(mode="keyword")
    oai      = OpenAI()

    print("=" * 60)
    print("  memory-os CLI — Ebbinghaus decay demo")
    print("  Type 'debug' to see memory scores")
    print("  Type 'quit' to exit")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "debug":
            raw     = store.retrieve("", top_k=50, update_access=False)
            ranked  = reranker.rerank(raw, top_n=10)
            print_memory_debug(ranked)
            continue

        # Retrieve + rerank
        raw     = store.retrieve(user_input, top_k=50)
        ranked  = reranker.rerank(raw, top_n=5)

        # Generate
        prompt   = build_prompt(user_input, ranked)
        response = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=300,
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
        print(f"\nAssistant: {reply}\n")

        # Store with emotion tag
        emotion = tagger.tag(user_input)
        importance = 0.6 if any(w in user_input.lower() for w in
                                ["prefer", "always", "never", "hate", "love",
                                 "important", "remember"]) else 0.4
        store.insert(
            text            = f"User: {user_input}",
            importance      = importance,
            emotional_score = emotion["score"],
            emotional_label = emotion["label"],
            source          = "cli_chat",
        )


if __name__ == "__main__":
    main()
