# memory-os

![CI](https://github.com/karann810/MemoryOS/actions/workflows/ci.yml/badge.svg)

Compact Qdrant-backed memory for AI applications.

MemoryOS does not generate final answers. Your application owns the chat flow and
the user-facing LLM call. MemoryOS extracts multiple important memory chunks from
the user's prompt, stores those chunks separately in Qdrant, applies
forgetting-curve style decay to them over time, and also keeps a small rolling
prompt/response history per user namespace.

MemoryOS also offers an explicit consolidation pass that a host app may invoke
from a background worker or scheduled job. That pass clusters vectors for one
user memory namespace and merges semantic duplicates into a single summarized
point.

## Public API

```python
pip install pymemoryos
from memory_os import MemoryOS

memory = MemoryOS(
    qdrant_url="https://your-qdrant-url",
    qdrant_api_key="your-qdrant-api-key",
    llm=configured_llm,                    # any object with .invoke()
)

memory.store(prompt: str, response: str, user_id: str | None = None) -> dict[str, bool]
memory.retrieve(prompt: str, user_id: str | None = None) -> dict
memory.consolidate(user_id: str) -> dict
```

That is the intended public surface, with `consolidate()` intentionally
called explicitly rather than automatically from the hot `store()` or
`retrieve()` paths. The package signals when consolidation is recommended
via `store()`, allowing the host app to trigger `consolidate(user_id)` asynchronously.

## Usage

```python
context = memory.retrieve(user_prompt, user_id="user_123")

final_response = host_llm.invoke(
    f"Relevant memory:\n{context}\n\nUser:\n{user_prompt}"
)

result = memory.store(user_prompt, final_response, user_id="user_123")
if result.get("consolidation_recommended"):
    # Host app schedules background worker for consolidation
    background_worker.enqueue(memory.consolidate, "user_123")
```

## Behavior

- `store(prompt, response, user_id=None)` calls `llm.invoke(...)` once to distill the completed
  prompt into multiple important memory chunks.
- `store(prompt, response, user_id=None)` returns `{"consolidation_recommended": bool}`, which becomes
  `True` every 20 stores for that specific user, signaling that consolidation should be run.
- `store(prompt, response, user_id=None)` embeds each extracted memory chunk with an internal
  SentenceTransformer model, then upserts those chunks separately to Qdrant.
- `store(prompt, response, user_id=None)` also stores the raw prompt/response pair in a simple
  rolling user history capped at 7 pairs.
- `retrieve(prompt, user_id=None)` embeds the current prompt with the same internal
  SentenceTransformer model, then queries Qdrant for relevant memories in that
  effective user namespace.
- `retrieve(prompt, user_id=None)` reranks Qdrant hits using an Ebbinghaus-style decay score:
  `similarity * decay_score * importance * emotional_weight`.
- `retrieve(prompt, user_id=None)` also returns the latest 4-5 stored prompt/response pairs for
  the same user namespace as immediate context.
- `consolidate(user_id)` is an explicit helper that scrolls Qdrant points for
  one user namespace with vectors, clusters similar points using cosine
  Agglomerative clustering, skips singletons, and replaces merged clusters with
  one summarized point whose payload carries merged facts like `merged_from`,
  recomputed `final_score`, and a `pair_id` reset to `None` so the merged point
  is outside the pair-based memory deletion lifecycle.
- All Qdrant reads and writes are filtered by `user_id`.
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
- MemoryOS consolidation uses `scikit-learn` for cosine AgglomerativeClustering.
- Required init config is exactly `qdrant_url`, `qdrant_api_key`, and `llm`.
