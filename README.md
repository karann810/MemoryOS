# memory-os

Compact Qdrant-backed memory for AI applications.

MemoryOS does not generate final answers. Your application owns the chat flow and
the user-facing LLM call. MemoryOS extracts multiple important memory chunks from
the user's prompt, stores those chunks separately in Qdrant, applies
forgetting-curve style decay to them over time, and also keeps a small rolling
prompt/response history per session.

## Public API

```python
pip install pymemoryos
from memory_os import MemoryOS

memory = MemoryOS(
    qdrant_url="https://your-qdrant-url",
    qdrant_api_key="your-qdrant-api-key",
    llm=configured_llm,                    # any object with .invoke()
    session_id="user_123",
)

memory.store(prompt: str, response: str) -> None
memory.retrieve(prompt: str) -> dict
```

That is the entire intended surface.

## Usage

```python
context = memory.retrieve(user_prompt)

final_response = host_llm.invoke(
    f"Relevant memory:\n{context}\n\nUser:\n{user_prompt}"
)

memory.store(user_prompt, final_response)
```

## Behavior

- `store(prompt, response)` calls `llm.invoke(...)` once to distill the completed
  prompt into multiple important memory chunks.
- `store(prompt, response)` embeds each extracted memory chunk with an internal
  SentenceTransformer model, then upserts those chunks separately to Qdrant.
- `store(prompt, response)` also stores the raw prompt/response pair in a simple
  rolling session history capped at 7 pairs.
- `retrieve(prompt)` embeds the current prompt with the same internal
  SentenceTransformer model, then queries Qdrant for relevant memories.
- `retrieve(prompt)` reranks Qdrant hits using an Ebbinghaus-style decay score:
  `similarity * decay_score * importance * emotional_weight`.
- `retrieve(prompt)` also returns the latest 4-5 stored prompt/response pairs for
  the same session as immediate context.
- All Qdrant reads and writes are filtered by `session_id`.
- When an old pair is evicted from the 7-pair history, the Qdrant memory chunks
  created from that prompt are also removed.
- Each stored memory chunk keeps decay state in Qdrant payload, including
  `last_accessed`, `access_count`, `stability`, `importance`, `emotion`,
  `emotional_weight`, `decay_score`, `final_score`, and `last_similarity`.

## Retrieve shape

```python
{
    "recent_pairs": [
        {"prompt": "...", "response": "...", "created_at": 1720000000.0},
    ],
    "memories": [
        {
            "text": "...",
            "score": 0.92,
            "created_at": 1720000000.0,
            "decay_score": 0.81,
            "importance": 0.9,
            "emotion": "fear",
        },
    ],
}
```

## Boundaries

- MemoryOS never generates the final answer to a user query.
- MemoryOS uses `llm.invoke()` only to break the user's prompt into storable memory chunks.
- MemoryOS uses an internal SentenceTransformer embedder for vector storage and retrieval.
- Required init config is exactly `qdrant_url`, `qdrant_api_key`, `llm`,
  and `session_id`.
