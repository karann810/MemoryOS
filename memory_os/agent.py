"""
memory_os/agent.py  —  MemoryAgent: context provider
========================================================
Extract, store, and retrieve relevant memories for user queries.
Works with any LLM provider (you handle the LLM call).

Minimal usage (6 params):

    from memory_os import MemoryAgent

    agent = MemoryAgent(
        llm_model      = "gpt-4o-mini",       # for fact extraction
        llm_api_key    = "sk-...",
        embed_model    = "text-embedding-3-small",
        embed_api_key  = "sk-...",            # often same as llm_api_key
        qdrant_url     = "http://localhost:6333",
        session_id     = "user_123",
    )
    
    # Get relevant memories
    contexts = agent.get_context("help with authentication")
    print(contexts)  # List of relevant memories
    
    # Use with your LLM
    formatted = "\\n".join([f"[{m['layer']}] {m['text']}" for m in contexts])
    response = your_llm.chat(system="...", user=f"{formatted}\\n\\n{message}")
"""

import os
from typing import Optional
try:
    import litellm
except Exception:
    from . import _litellm as litellm

litellm.suppress_debug_info = True

from ._utils import set_litellm_key

# LiteLLM is used for:
# - Memory extraction (identifying facts from user messages)
# - Emotion tagging (optional)


class MemoryAgent:
    """
    Context provider for persistent user memory.

    Handles memory extraction, storage, ranking, and retrieval.
    All features intact: vector embeddings, decay, emotions, extraction.
    
    LiteLLM is used internally for fact extraction and emotion tagging.
    YOU handle the LLM response generation with the contexts returned.

    Required parameters:
    ────────────────────
    llm_model      : LiteLLM model string (e.g. "gpt-4o-mini", "claude-3-haiku-20240307")
    llm_api_key    : API key for the LLM provider
    embed_model    : Embedding model (e.g. "text-embedding-3-small")
    embed_api_key  : API key for embeddings (often same as llm_api_key)
    qdrant_url     : Qdrant URL ("http://localhost:6333" or Qdrant Cloud URL)
    session_id     : Unique ID per user/session

    Optional parameters:
    ────────────────────
    qdrant_api_key  : Qdrant Cloud API key (None for local Qdrant)
    top_n_memories  : Memories injected per prompt (default: 5)
    emotion_mode    : "llm" = tag emotions (accurate) | "off" = skip (faster)
    """

    def __init__(
        self,
        llm_model:      str,
        llm_api_key:    str,
        embed_model:    str,
        embed_api_key:  str,
        qdrant_url:     str,
        session_id:     str,
        # Optional
        qdrant_api_key:  Optional[str] = None,
        top_n_memories:  int = 5,
        emotion_mode:    str = "llm",  # "llm" = tag emotions (accurate) | "off" = skip (faster)
        redis_url:       Optional[str] = None,
    ):
        missing = [k for k, v in {
            "llm_model":     llm_model,
            "llm_api_key":   llm_api_key,
            "embed_model":   embed_model,
            "embed_api_key": embed_api_key,
            "qdrant_url":    qdrant_url,
            "session_id":    session_id,
        }.items() if not v]
        if missing:
            raise ValueError(
                f"Missing required parameters: {missing}\n\n"
                "Minimum usage:\n"
                "    agent = MemoryAgent(\n"
                "        llm_model     = 'gpt-4o-mini',\n"
                "        llm_api_key   = 'sk-...',\n"
                "        embed_model   = 'text-embedding-3-small',\n"
                "        embed_api_key = 'sk-...',\n"
                "        qdrant_url    = 'http://localhost:6333',\n"
                "        session_id    = 'user_1',\n"
                "    )"
            )

        self.llm_model      = llm_model
        self.session_id     = session_id
        self.top_n_memories = top_n_memories

        # Set API keys for litellm
        set_litellm_key(llm_model, llm_api_key)
        if embed_api_key != llm_api_key:
            set_litellm_key(embed_model, embed_api_key)

        # Init components
        from .store     import MemoryStore
        from .decay     import DecayReranker
        from .emotion   import EmotionTagger
        from .extractor import MemoryExtractor
        from .router    import MemoryRouter

        self.store = MemoryStore(
            qdrant_url    = qdrant_url,
            qdrant_key    = qdrant_api_key,
            embed_model   = embed_model,
            embed_api_key = embed_api_key,
        )
        self.reranker  = DecayReranker()
        self.tagger    = EmotionTagger(model=llm_model, api_key=llm_api_key, mode=emotion_mode)
        self.extractor = MemoryExtractor(model=llm_model, api_key=llm_api_key)
        self.router    = MemoryRouter(
            store          = self.store,
            decay_reranker = self.reranker,
            session_id     = session_id,
            redis_url      = redis_url,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def get_context(self, message: str, recent_n: int | None = None) -> list[dict]:
        """
        Get relevant memories for a user query (context provider).
        
        Extracts facts, retrieves ranked memories, and stores the interaction.
        You handle the LLM call with this context.

        Args:
            message: User query/message
            
        Returns:
            list[dict]: Relevant memories ranked by relevance. Each contains:
                - text: Memory content
                - layer: Memory type (semantic/episodic/semantic)
                - importance: Score 0-1
                - emotional_label: Emotion tag (if emotion_mode="llm")

        Usage:
            contexts = agent.get_context("help with authentication")
            formatted = "\n".join([f"[{m['layer']}] {m['text']}" for m in contexts])
            
            # Use with your LLM
            response = my_llm.chat(
                system="You are helpful.",
                user=f"{formatted}\n\nUser's message: {message}"
            )
        """
        facts    = self.extractor.extract(message)
        memories = self.router.query(message, top_n=self.top_n_memories)

        # Optionally include recent user/assistant pairs from working memory
        if recent_n and recent_n > 0:
            pairs = self.router.get_recent_pairs(recent_n)
            paired_memories = []
            for p in pairs:
                text = f"User: {p['user']}\nAssistant: {p['assistant']}"
                paired_memories.append({"text": text, "score": 0.9, "layer": "working_pair"})
            memories = paired_memories + memories

        # Do NOT store facts or interactions here — caller will call
        # `store()` after getting the LLM response.
        return memories
        
    def store(
        self,
        message: str,
        response: str,
        extraction_result = None,
        source: str = "interaction",
    ) -> dict:
        """Public method to store a completed interaction.

        Call this after you have used `get_context()` and obtained an LLM
        response. This will:
          - push the user message and assistant response into working memory
          - extract and store facts from the user message
          - store the query/response pair as memories

        Returns a dict with `fact_ids` and `interaction_ids`.
        """
        facts = extraction_result if extraction_result is not None else self.extractor.extract(message)

        # Push both to working memory (user then assistant)
        self.router.push_to_working(message, role="user")
        self.router.push_to_working(response, role="assistant")

        # Store extracted facts
        fact_ids = []
        if getattr(facts, "has_facts", False):
            fact_ids = self.extractor.extract_and_store(
                text=message,
                store=self.store,
                user_id=self.session_id,
                source=f"session:{self.session_id}",
                _result=facts,
            )

        # Store the query/response pair
        interaction_ids = self.store.store_interaction(
            user_id = self.session_id,
            query = message,
            response = response,
            source = f"session:{self.session_id}",
        )

        return {"fact_ids": fact_ids, "interaction_ids": interaction_ids}

    # ── Internals ─────────────────────────────────────────────────────────────

    def _store_facts(self, message: str, extraction_result) -> list[str]:
        """Store extracted facts with emotion tags."""
        if getattr(extraction_result, "has_facts", False):
            return self.extractor.extract_and_store(
                text    = message,
                store   = self.store,
                user_id = self.session_id,
                source  = f"session:{self.session_id}",
                _result = extraction_result,
            )
        else:
            emotion = self.tagger.tag(message)
            mid = self.store.insert(
                user_id         = self.session_id,
                text            = f"User said: {message}",
                importance      = 0.3,
                memory_type     = "episodic",
                emotional_score = emotion.score,
                emotional_label = emotion.label,
                source          = f"session:{self.session_id}",
            )
            return [mid]
