"""
memory_os/exceptions.py  —  All custom exceptions for memory-os
===============================================================
Three categories:

  1. Setup errors    — user misconfigured something, fail immediately
  2. Runtime errors  — something broke mid-conversation, degrade gracefully  
  3. Data errors     — bad data in the system, handle and continue

Usage:
    from memory_os.exceptions import (
        MemoryOSError,
        MemoryOSAuthError,
        MemoryOSConnectionError,
        MemoryOSModelError,
    )

    try:
        agent = MemoryAgent(...)
    except MemoryOSAuthError as e:
        print(f"Bad API key: {e}")
    except MemoryOSConnectionError as e:
        print(f"Qdrant not running: {e}")
    except MemoryOSError as e:
        print(f"Something went wrong: {e}")
"""


# ── Base exception — catch all memory-os errors with this ────────────────────

class MemoryOSError(Exception):
    """
    Base class for all memory-os exceptions.
    Catch this if you want to handle any memory-os error in one place.

        try:
            contexts = agent.get_context("hello")
        except MemoryOSError as e:
            print(e)
    """
    def __init__(self, message: str, original_error: Exception = None):
        self.original_error = original_error
        super().__init__(message)


# ── Category 1: Setup errors ─────────────────────────────────────────────────
# These happen at startup. Fail immediately with clear message.

class MemoryOSAuthError(MemoryOSError):
    """
    Wrong or missing API key.

    When it happens:
      - openai_api_key is wrong
      - embed_api_key is wrong
      - Qdrant API key rejected

    Example:
        MemoryAgent(openai_api_key="sk-wrong", ...)
        → MemoryOSAuthError: OpenAI API key is invalid.
          Check your key at https://platform.openai.com/api-keys
    """
    pass


class MemoryOSModelError(MemoryOSError):
    """
    Wrong model name or incompatible model for the task.

    When it happens:
      - Model name doesn't exist ("gpt-5-ultra")
      - Using a Groq model for embeddings (Groq has no embeddings API)
      - Model string is malformed

    Example:
        MemoryAgent(embed_model="groq/llama-3.1-8b", ...)
        → MemoryOSModelError: groq/llama-3.1-8b cannot be used for embeddings.
          Groq does not have an embeddings API.
          Use: gemini/text-embedding-004 (free) or text-embedding-3-small (OpenAI)
    """
    pass


class MemoryOSConnectionError(MemoryOSError):
    """
    Cannot connect to a vector database.

    When it happens:
      - Qdrant not running at the given URL
      - Weaviate cluster unreachable
      - Pinecone index doesn't exist
      - Network timeout on connection

    Example:
        MemoryAgent(qdrant_url="http://localhost:6333", ...)
        → MemoryOSConnectionError: Cannot connect to Qdrant at http://localhost:6333
          Is Qdrant running? Start it with:
          docker run -p 6333:6333 qdrant/qdrant
    """
    pass


class MemoryOSConfigError(MemoryOSError):
    """
    Bad configuration — missing required fields, incompatible settings.

    When it happens:
      - session_id not provided
      - vector_dim mismatch with existing Qdrant collection
      - Both llm_api_key and embed_api_key missing

    Example:
        MemoryAgent(openai_api_key="sk-...", session_id="")
        → MemoryOSConfigError: session_id cannot be empty.
          Use a unique string per user, e.g. session_id="user_123"
    """
    pass


# ── Category 2: Runtime errors ───────────────────────────────────────────────
# These happen mid-conversation. Agent degrades gracefully instead of crashing.

class MemoryOSRateLimitError(MemoryOSError):
    """
    API rate limit hit. Agent will retry automatically with backoff.

    When it happens:
      - Too many requests to OpenAI/Gemini/Groq
      - Free tier limit reached

    The agent catches this internally and retries.
    Only raised to the user if all retries fail.
    """
    def __init__(self, provider: str, retry_after: int = None, **kwargs):
        self.provider    = provider
        self.retry_after = retry_after
        msg = f"Rate limit hit on {provider}."
        if retry_after:
            msg += f" Retry after {retry_after} seconds."
        super().__init__(msg, **kwargs)


class MemoryOSTimeoutError(MemoryOSError):
    """
    API request timed out.

    When it happens:
      - LLM took too long to respond
      - Qdrant search timed out on large collections
    """
    pass


class MemoryOSStorageError(MemoryOSError):
    """
    Failed to store or retrieve a memory.

    When it happens:
      - Qdrant write failed
      - Vector dimension mismatch on insert
      - Qdrant collection was deleted externally

    Agent catches this and continues — the conversation keeps working,
    just that specific memory won't be stored.
    """
    pass


# ── Category 3: Data errors ──────────────────────────────────────────────────
# Bad data in the system. Handled internally, rarely raised to user.

class MemoryOSExtractionError(MemoryOSError):
    """
    Memory extraction failed — GPT returned invalid JSON or empty response.
    Agent falls back to storing the whole message as one memory.
    """
    pass


class MemoryOSEmotionError(MemoryOSError):
    """
    Emotion tagging failed.
    Agent falls back to storing memory as neutral emotion.
    """
    pass


# ── Helper: wrap external errors into memory-os errors ───────────────────────

def wrap_litellm_error(error: Exception, context: str = "") -> MemoryOSError:
    """
    Convert LiteLLM/OpenAI errors into clean memory-os exceptions
    with helpful messages.

    Usage:
        try:
            litellm.completion(...)
        except Exception as e:
            raise wrap_litellm_error(e, context="emotion tagging")
    """
    error_str = str(error).lower()
    error_type = type(error).__name__

    # Auth errors
    if any(x in error_str for x in ["invalid api key", "unauthorized",
                                      "authentication", "401", "api key"]):
        return MemoryOSAuthError(
            f"API key rejected during {context}.\n"
            f"Check your llm_api_key or embed_api_key.\n"
            f"Original error: {error}",
            original_error=error,
        )

    # Rate limits
    if any(x in error_str for x in ["rate limit", "429", "too many requests",
                                      "quota exceeded"]):
        return MemoryOSRateLimitError(
            provider=_detect_provider(error_str),
            original_error=error,
        )

    # Model not found
    if any(x in error_str for x in ["model not found", "invalid model",
                                      "does not exist", "404", "no such model"]):
        return MemoryOSModelError(
            f"Model not found during {context}.\n"
            f"Check your llm_model or embed_model string.\n"
            f"Original error: {error}",
            original_error=error,
        )

    # Timeout
    if any(x in error_str for x in ["timeout", "timed out", "deadline exceeded"]):
        return MemoryOSTimeoutError(
            f"Request timed out during {context}.",
            original_error=error,
        )

    # Connection errors
    if any(x in error_str for x in ["connection", "network", "refused",
                                      "unreachable", "502", "503"]):
        return MemoryOSConnectionError(
            f"Connection failed during {context}.\n"
            f"Original error: {error}",
            original_error=error,
        )

    # Unknown — wrap as base error
    return MemoryOSError(
        f"Unexpected error during {context}: {error}",
        original_error=error,
    )


def wrap_qdrant_error(error: Exception, qdrant_url: str = "") -> MemoryOSError:
    """Convert Qdrant errors into clean memory-os exceptions."""
    error_str = str(error).lower()

    if any(x in error_str for x in ["connection refused", "failed to connect",
                                      "unreachable", "timeout"]):
        return MemoryOSConnectionError(
            f"Cannot connect to Qdrant at {qdrant_url}\n"
            f"Is Qdrant running? Start it with:\n"
            f"  docker run -p 6333:6333 qdrant/qdrant\n"
            f"Original error: {error}",
            original_error=error,
        )

    if "unauthorized" in error_str or "api key" in error_str:
        return MemoryOSAuthError(
            f"Qdrant API key rejected.\n"
            f"Check your qdrant_api_key.\n"
            f"Original error: {error}",
            original_error=error,
        )

    return MemoryOSStorageError(
        f"Qdrant operation failed: {error}",
        original_error=error,
    )


def _detect_provider(error_str: str) -> str:
    if "openai" in error_str:    return "OpenAI"
    if "gemini" in error_str:    return "Gemini"
    if "groq" in error_str:      return "Groq"
    if "anthropic" in error_str: return "Anthropic"
    if "cohere" in error_str:    return "Cohere"
    return "LLM provider"
