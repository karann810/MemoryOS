"""
memory_os/validators.py  —  Startup validation
===============================================
Runs before the agent starts. Catches every misconfiguration
immediately with a clear, helpful error message.

Philosophy:
  Fail loudly at startup with a helpful message
  rather than silently failing 10 messages in.

  Like a car dashboard — if the engine light turns on,
  tell the driver NOW, not after 10 miles.
"""

import os
from .exceptions import (
    MemoryOSAuthError,
    MemoryOSModelError,
    MemoryOSConnectionError,
    MemoryOSConfigError,
)

# ── Models that CANNOT be used for embeddings ────────────────────────────────
# These providers have no embeddings API.

NO_EMBEDDING_PROVIDERS = [
    "groq",
    "xai",
    "anthropic",
    "perplexity",
    "together",
]

# ── Known valid embedding model prefixes ─────────────────────────────────────

VALID_EMBEDDING_MODELS = [
    "text-embedding",           # OpenAI: text-embedding-3-small etc
    "gemini/text-embedding",    # Gemini: gemini/text-embedding-004
    "cohere/embed",             # Cohere: cohere/embed-english-v3.0
    "voyage",                   # Voyage AI
    "mistral/mistral-embed",    # Mistral
]

# ── Vector dimensions per embedding model ────────────────────────────────────

EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small":        1536,
    "text-embedding-3-large":        3072,
    "text-embedding-ada-002":        1536,
    "gemini/text-embedding-004":     768,
    "cohere/embed-english-v3.0":     1024,
    "cohere/embed-multilingual-v3.0": 1024,
    "mistral/mistral-embed":         1024,
}


def validate_all(
    llm_api_key:   str,
    embed_api_key: str,
    llm_model:     str,
    embed_model:   str,
    session_id:    str,
    qdrant_url:    str,
    qdrant_api_key: str = None,
) -> int:
    """
    Run all startup validations.
    Returns the correct vector_dim for the chosen embed_model.
    Raises a specific MemoryOSError subclass on any problem.
    """
    validate_session_id(session_id)
    validate_api_keys(llm_api_key, embed_api_key)
    validate_embed_model(embed_model)
    validate_qdrant_connection(qdrant_url, qdrant_api_key)
    vector_dim = get_vector_dim(embed_model)
    return vector_dim


# ── Individual validators ─────────────────────────────────────────────────────

def validate_session_id(session_id: str) -> None:
    if not session_id or not session_id.strip():
        raise MemoryOSConfigError(
            "session_id cannot be empty.\n"
            "Use a unique string per user, e.g:\n"
            "  MemoryAgent(session_id='user_123', ...)\n"
            "  MemoryAgent(session_id='karan_dev', ...)"
        )
    if len(session_id) > 100:
        raise MemoryOSConfigError(
            f"session_id is too long ({len(session_id)} chars). Max 100 characters."
        )
    # Check for characters that break Pinecone namespaces
    invalid_chars = set('/ \\ " \' < > { } | ^ ` \n \t')
    found = [c for c in session_id if c in invalid_chars]
    if found:
        raise MemoryOSConfigError(
            f"session_id contains invalid characters: {found}\n"
            "Use only letters, numbers, hyphens, underscores.\n"
            "e.g. 'user_123' or 'karan-dev'"
        )


def validate_api_keys(llm_api_key: str, embed_api_key: str) -> None:
    if not llm_api_key or not llm_api_key.strip():
        raise MemoryOSAuthError(
            "llm_api_key is required.\n\n"
            "Get a free key from one of these:\n"
            "  Groq (free):   https://console.groq.com\n"
            "  Gemini (free): https://aistudio.google.com/apikey\n"
            "  OpenAI:        https://platform.openai.com/api-keys\n\n"
            "Then pass it:\n"
            "  MemoryAgent(llm_api_key='your-key', ...)"
        )

    if not embed_api_key or not embed_api_key.strip():
        raise MemoryOSAuthError(
            "embed_api_key is required.\n\n"
            "For free embeddings use Gemini:\n"
            "  embed_model   = 'gemini/text-embedding-004'\n"
            "  embed_api_key = 'your-gemini-key'\n\n"
            "Get a free Gemini key: https://aistudio.google.com/apikey"
        )

    # Basic format checks — catch obvious typos
    _check_key_format(llm_api_key, "llm_api_key")
    if embed_api_key != llm_api_key:
        _check_key_format(embed_api_key, "embed_api_key")


def validate_embed_model(embed_model: str) -> None:
    if not embed_model:
        raise MemoryOSModelError(
            "embed_model is required.\n\n"
            "Choose one:\n"
            "  'gemini/text-embedding-004'  ← free\n"
            "  'text-embedding-3-small'     ← OpenAI, best quality\n"
            "  'cohere/embed-english-v3.0'  ← free tier"
        )

    # Check if using an LLM-only provider for embeddings
    model_lower = embed_model.lower()
    for provider in NO_EMBEDDING_PROVIDERS:
        if model_lower.startswith(provider):
            _raise_no_embedding_error(provider, embed_model)

    # Warn if model looks like a chat model, not an embedding model
    chat_model_hints = ["gpt-4", "gpt-3.5", "claude", "gemini-1.5",
                        "llama", "mistral-7b", "mixtral", "grok"]
    for hint in chat_model_hints:
        if hint in model_lower and "embed" not in model_lower:
            raise MemoryOSModelError(
                f"'{embed_model}' looks like a chat model, not an embedding model.\n\n"
                f"Chat models cannot convert text to vectors.\n"
                f"Use an embedding model instead:\n"
                f"  'gemini/text-embedding-004'  ← free\n"
                f"  'text-embedding-3-small'     ← OpenAI\n"
                f"  'cohere/embed-english-v3.0'  ← free tier"
            )


def validate_qdrant_connection(qdrant_url: str, qdrant_api_key: str = None) -> None:
    """
    Try to connect to Qdrant. Raises MemoryOSConnectionError if it fails.
    """
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=5)
        client.get_collections()  # lightweight health check
    except Exception as e:
        error_str = str(e).lower()
        if "unauthorized" in error_str or "api key" in error_str:
            raise MemoryOSAuthError(
                f"Qdrant rejected the API key at {qdrant_url}\n"
                f"Check your qdrant_api_key.",
                original_error=e,
            )
        raise MemoryOSConnectionError(
            f"Cannot connect to Qdrant at {qdrant_url}\n\n"
            f"Is Qdrant running? Start it with:\n"
            f"  docker run -p 6333:6333 qdrant/qdrant\n\n"
            f"Or use Qdrant Cloud: https://cloud.qdrant.io\n"
            f"Original error: {e}",
            original_error=e,
        )


def get_vector_dim(embed_model: str) -> int:
    """
    Return the correct vector dimension for the given embedding model.
    Qdrant collection must be created with this dimension.
    """
    # Exact match first
    if embed_model in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[embed_model]

    # Partial match
    model_lower = embed_model.lower()
    if "gemini" in model_lower and "embedding" in model_lower:
        return 768
    if "text-embedding-3-large" in model_lower:
        return 3072
    if "text-embedding" in model_lower:
        return 1536
    if "cohere" in model_lower:
        return 1024
    if "mistral" in model_lower and "embed" in model_lower:
        return 1024

    # Unknown model — default to 1536 with a warning
    import warnings
    warnings.warn(
        f"Unknown embedding model '{embed_model}'. "
        f"Defaulting to vector_dim=1536. "
        f"Pass vector_dim= explicitly if this is wrong.",
        UserWarning,
        stacklevel=3,
    )
    return 1536


# ── Private helpers ───────────────────────────────────────────────────────────

def _check_key_format(key: str, key_name: str) -> None:
    """Catch obviously wrong key formats."""
    if key.startswith("your-") or key in ("sk-...", "AIza...", "gsk_..."):
        raise MemoryOSAuthError(
            f"{key_name} looks like a placeholder, not a real key.\n"
            f"Replace '{key}' with your actual API key."
        )
    if len(key) < 10:
        raise MemoryOSAuthError(
            f"{key_name} is too short to be a valid API key.\n"
            f"Check you copied the full key."
        )


def _raise_no_embedding_error(provider: str, model: str) -> None:
    suggestions = {
        "groq":      "You can still use Groq for llm_model.\n"
                     "Just use a different provider for embed_model.",
        "anthropic": "You can still use Anthropic for llm_model.\n"
                     "Just use a different provider for embed_model.",
        "xai":       "xAI (Grok) does not have an embeddings API yet.",
    }
    note = suggestions.get(provider, f"{provider} does not support embeddings.")
    raise MemoryOSModelError(
        f"'{model}' cannot be used for embeddings.\n"
        f"{note}\n\n"
        f"Use one of these for embed_model:\n"
        f"  'gemini/text-embedding-004'  ← free\n"
        f"  'text-embedding-3-small'     ← OpenAI, best quality\n"
        f"  'cohere/embed-english-v3.0'  ← free tier\n\n"
        f"Example:\n"
        f"  MemoryAgent(\n"
        f"    llm_model     = 'groq/llama-3.1-8b-instant',\n"
        f"    llm_api_key   = 'gsk_...',\n"
        f"    embed_model   = 'gemini/text-embedding-004',\n"
        f"    embed_api_key = 'AIza...',\n"
        f"  )"
    )
