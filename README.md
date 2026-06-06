# memory-os

**Brain-like memory for AI agents.**

Unlike Mem0, Zep, and LangChain Memory — memory-os is grounded in
published psychology research. It forgets like a human, not like a database.

```python
from memory_os import MemoryAgent

agent = MemoryAgent(session_id="user_123")
response = agent.chat("I hate verbose explanations")
# Later...
response = agent.chat("Explain how Python decorators work")
# Agent remembers your preference and keeps it concise
```

## What makes it different

Every other tool ranks memories by cosine similarity alone. We don't.

**Ebbinghaus forgetting curve** (1885):
```
R(t, S) = e^(-t/S)
```
Memory retention decays exponentially with time, but stability `S` grows
every time a memory is successfully retrieved with proper spacing.
Old, unaccessed memories fade. Frequently reviewed memories persist.

**Emotional weighting** (McGaugh, 2000):
Emotionally charged events (fear, anger, joy) are remembered 1.5-2x
longer than neutral ones. Your agent treats "I lost all my code and
nearly failed the project" differently from "I had coffee this morning."

**Memory consolidation** (Gap 2 — unpublished):
A nightly background job clusters episodic memories and compresses them
into semantic insights — exactly like sleep consolidates human memory.
We measure and log whether the summary vector sits at the cluster centroid.
This finding is publishable.

**Final retrieval score:**
```
M = similarity × R(t,S) × importance × emotional_weight
```

## Architecture

```
Working memory  →  Pinecone    (last 15 msgs, ultra-low latency)
Episodic memory →  Qdrant      (time-scored events, Ebbinghaus decay)
Semantic memory →  Weaviate    (compressed summaries, hybrid search)
```

## Installation

```bash
pip install memory-os

# Start Qdrant locally
docker run -p 6333:6333 qdrant/qdrant

# Copy and fill in your API keys
cp .env.example .env
```

## Quick start

```python
from memory_os import MemoryAgent

agent = MemoryAgent(session_id="alice")

# Chat — memories auto-stored with emotion tagging
agent.chat("I'm terrified of breaking production.")
agent.chat("I always prefer typed Python over untyped.")

# Later — correct memories surface, old noise fades
response = agent.chat("Should I add type hints to this function?")

# End session — flush working memory
agent.end_session()
```

## Run the benchmark

```bash
python benchmark/run_benchmark.py
```

Compares memory-os vs vanilla RAG across 4 tasks.
Results saved to `benchmark/results.md`.

## Run the CLI demo

```bash
python scripts/cli_chat.py
```

Have 10 conversations. Type `debug` to see Ebbinghaus scores live.

## Run tests

```bash
pytest tests/ -v
```

## Research references

- Ebbinghaus, H. (1885). *Über das Gedächtnis*. Leipzig: Duncker & Humblot.
- McGaugh, J.L. (2000). Memory — a century of consolidation. *Science*, 287(5451), 248-251.
- Wozniak & Gorzelanczyk (1994). Optimization of repetition spacing in the expansion-rehearsal system.

## License

MIT
