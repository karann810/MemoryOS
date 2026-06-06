"""
memory_os/agent.py  —  LangGraph Agent (updated with extractor)
===============================================================
Updated flow:

  user message
    ↓
  EXTRACT   — pull out N distinct facts from the message
    ↓
  RETRIEVE  — search all 3 memory layers for relevant context
    ↓
  GENERATE  — LLM answers using retrieved memories as context
    ↓
  MEMORIZE  — store extracted facts (each separately, emotion-tagged)
    ↓
  response

The key change from v1:
  Before: store whole message as one blob
  Now:    extract N facts, store each with own importance + emotion + decay
"""

import os
from typing import Optional, TypedDict
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYSTEM_PROMPT = """You are a helpful AI assistant with persistent memory.
You remember past conversations and use them to give better, more personalized responses.

When relevant memories are provided, use them naturally — don't announce "I remember that..."
Just incorporate the context as a knowledgeable friend would.

Be concise and helpful."""

CONTEXT_TEMPLATE = """Relevant memories about this user:
{memories}

User's message: {query}"""


class AgentState(TypedDict):
    query:           str
    extracted_facts: list[dict]   # what we learned from this message
    memories:        list[dict]   # what we retrieved for context
    context_prompt:  str
    response:        str
    stored_ids:      list[str]    # UUIDs of newly stored memories


class MemoryAgent:
    """
    Full memory-os agent with extraction, retrieval, generation, storage.

    Usage:
        agent = MemoryAgent(session_id="user_123")
        response = agent.chat("I'm building with Next.js, help me set up auth")
        print(response)

        # With debug info
        result = agent.chat_with_debug("same message")
        print(result["extracted_facts"])   # what was extracted
        print(result["memories_used"])     # what memories were retrieved
    """

    def __init__(
        self,
        session_id:      str = "default",
        openai_key:      Optional[str] = None,
        model:           str = "gpt-4o-mini",
        use_langgraph:   bool = True,
        top_n_memories:  int = 5,
    ):
        from .store      import MemoryStore
        from .decay      import DecayReranker
        from .emotion    import EmotionTagger
        from .router     import MemoryRouter
        from .extractor  import MemoryExtractor

        self.session_id     = session_id
        self.model          = model
        self.top_n_memories = top_n_memories
        self._oai           = OpenAI(api_key=openai_key or os.getenv("OPENAI_API_KEY"))

        # Core components
        self.store     = MemoryStore()
        self.reranker  = DecayReranker()
        self.tagger    = EmotionTagger()
        self.router    = MemoryRouter(
            store          = self.store,
            decay_reranker = self.reranker,
            session_id     = session_id,
        )
        self.extractor = MemoryExtractor()

        if use_langgraph:
            self._graph = self._build_graph()
        else:
            self._graph = None

    # ── LangGraph graph ───────────────────────────────────────────────────────

    def _build_graph(self):
        try:
            from langgraph.graph import StateGraph, END
        except ImportError:
            raise ImportError("pip install langgraph")

        graph = StateGraph(AgentState)

        # 4 nodes now — extract is new
        graph.add_node("extract",   self._node_extract)
        graph.add_node("retrieve",  self._node_retrieve)
        graph.add_node("generate",  self._node_generate)
        graph.add_node("memorize",  self._node_memorize)

        graph.set_entry_point("extract")
        graph.add_edge("extract",  "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "memorize")
        graph.add_edge("memorize", END)

        return graph.compile()

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def _node_extract(self, state: AgentState) -> AgentState:
        """
        Node 1 — Extract distinct facts from the user's message.

        "I'm building with Next.js, hate Firebase, 3 week deadline"
        becomes:
        [
          {text: "User builds with Next.js", importance: 0.9, type: "semantic"},
          {text: "User hates Firebase",       importance: 0.8, type: "semantic"},
          {text: "User has 3 week deadline",  importance: 0.7, type: "episodic"},
        ]
        """
        facts = self.extractor.extract(state["query"])
        return {**state, "extracted_facts": facts}

    def _node_retrieve(self, state: AgentState) -> AgentState:
        """
        Node 2 — Retrieve relevant memories for this query.
        Also pushes raw message to working memory.
        """
        query    = state["query"]
        memories = self.router.query(query, top_n=self.top_n_memories)

        # Push to working memory (raw message, not extracted)
        self.router.push_to_working(query, role="user")

        # Build context prompt
        if memories:
            mem_lines = "\n".join(
                f"[{m['layer']}] {m['text']}" for m in memories
            )
            context = CONTEXT_TEMPLATE.format(
                memories=mem_lines,
                query=query,
            )
        else:
            context = query

        return {**state, "memories": memories, "context_prompt": context}

    def _node_generate(self, state: AgentState) -> AgentState:
        """Node 3 — Generate response using LLM + retrieved memories."""
        response = self._oai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": state["context_prompt"]},
            ],
            temperature=0.7,
            max_tokens=500,
        )
        reply = response.choices[0].message.content.strip()
        self.router.push_to_working(reply, role="assistant")
        return {**state, "response": reply}

    def _node_memorize(self, state: AgentState) -> AgentState:
        """
        Node 4 — Store extracted facts as separate memories.

        Each fact gets:
        - Its own importance score (from extractor)
        - Its own emotion tag (from tagger, per-fact not per-message)
        - Its own Ebbinghaus decay rate (based on above two)
        - Its own memory_type (episodic or semantic)
        """
        stored_ids = []

        if state["extracted_facts"]:
            # Store each extracted fact individually
            ids = self.extractor.extract_and_store(
                text   = state["query"],
                store  = self.store,
                tagger = self.tagger,
                source = f"session:{self.session_id}",
            )
            stored_ids.extend(ids)
        else:
            # Fallback: if nothing extracted, store the whole message
            # (handles short messages, greetings, etc.)
            emotion = self.tagger.tag(state["query"])
            mid = self.store.insert(
                text            = f"User said: {state['query']}",
                importance      = 0.3,
                memory_type     = "episodic",
                emotional_score = emotion["score"],
                emotional_label = emotion["label"],
                source          = f"session:{self.session_id}",
            )
            stored_ids.append(mid)

        return {**state, "stored_ids": stored_ids}

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(self, query: str) -> str:
        """Send message, get response. Memories handled automatically."""
        if self._graph:
            final = self._graph.invoke({
                "query":           query,
                "extracted_facts": [],
                "memories":        [],
                "context_prompt":  "",
                "response":        "",
                "stored_ids":      [],
            })
            return final["response"]
        return self._simple_chat(query)

    def chat_with_debug(self, query: str) -> dict:
        """
        Like chat() but returns full pipeline info.
        Shows exactly what was extracted, what memories were used,
        and what was stored. Perfect for debugging + benchmark.
        """
        final = self._graph.invoke({
            "query":           query,
            "extracted_facts": [],
            "memories":        [],
            "context_prompt":  "",
            "response":        "",
            "stored_ids":      [],
        })
        return {
            "response":        final["response"],
            "extracted_facts": final["extracted_facts"],  # what we learned
            "memories_used":   final["memories"],         # what we retrieved
            "stored_ids":      final["stored_ids"],       # new memory UUIDs
            "query":           query,
        }

    def end_session(self) -> None:
        """Flush working memory at end of session."""
        self.router.flush_working()

    def _simple_chat(self, query: str) -> str:
        """Fallback without LangGraph."""
        facts    = self.extractor.extract(query)
        memories = self.router.query(query, top_n=self.top_n_memories)
        mem_lines = "\n".join(f"- {m['text']}" for m in memories)
        context  = CONTEXT_TEMPLATE.format(
            memories=mem_lines, query=query
        ) if memories else query

        response = self._oai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": context},
            ],
            temperature=0.7,
            max_tokens=500,
        )
        reply = response.choices[0].message.content.strip()

        # Store extracted facts
        if facts:
            for fact in facts:
                emotion = self.tagger.tag(fact["text"])
                self.store.insert(
                    text            = fact["text"],
                    importance      = fact["importance"],
                    memory_type     = fact["memory_type"],
                    emotional_score = emotion["score"],
                    emotional_label = emotion["label"],
                    source          = f"session:{self.session_id}",
                )
        return reply
