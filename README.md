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
Working memory  →  Qdrant      (user-scoped recent prompt/response history)
Episodic memory →  Qdrant      (time-scored events, Ebbinghaus decay)
```

## Installation

```bash
pip install memory-os

# Qdrant cloud mode requires URL + API key
e.g. QDRANT_URL=https://your-project.a0c6f4.qdrant.cloud
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION=memories

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

## Two-function API (minimal surface)

If you prefer to manage LLM calls yourself, use the two simple methods on `MemoryAgent`:

- `get_context(prompt: str) -> dict` — read-only retrieval that returns a `context_prompt` ready to include in your LLM input and a `memories` list. This does NOT update access metadata or push to working memory.
- `store_interaction(prompt: str, response: str) -> list[str]` — after you receive the LLM response, call this to push the prompt and response into Qdrant working memory and to extract + store important facts from the prompt. Returns stored memory UUIDs.

Example:

```python
from memory_os import MemoryAgent

agent = MemoryAgent(session_id="alice", use_langgraph=False)

# 1) Get context to include in your LLM call (read-only)
ctx = agent.get_context("How do I set up auth in Next.js?")
# Use ctx['context_prompt'] when building your LLM messages

# 2) You call your LLM with the prompt + ctx['context_prompt'] and receive `llm_reply`

# 3) Store the interaction and extracted memories
stored_ids = agent.store_interaction("How do I set up auth in Next.js?", llm_reply)
print("Stored memory ids:", stored_ids)
```

Notes:
- `get_context` is intentionally read-only so retrieval doesn't alter memory access traces unless you explicitly call `store_interaction`.
- Extraction runs on the user's `prompt` only (not the assistant response) by default; extracted facts are emotion-tagged and stored individually.


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
