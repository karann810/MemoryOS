"""
memory_os/_utils.py — shared utilities (avoids circular imports)
"""
import os


def set_litellm_key(model: str, api_key: str) -> None:
    """Set the correct environment variable for the LLM provider."""
    if not api_key:
        return
    m = model.lower()
    if "claude" in m or "anthropic" in m:
        os.environ["ANTHROPIC_API_KEY"] = api_key
    elif "gemini" in m or "google" in m:
        os.environ["GEMINI_API_KEY"] = api_key
        os.environ["GOOGLE_API_KEY"] = api_key
    elif "groq" in m:
        os.environ["GROQ_API_KEY"] = api_key
    elif "cohere" in m:
        os.environ["COHERE_API_KEY"] = api_key
    elif "ollama" in m:
        pass  # local, no key
    elif "mistral" in m:
        os.environ["MISTRAL_API_KEY"] = api_key
    elif "together" in m:
        os.environ["TOGETHERAI_API_KEY"] = api_key
    else:
        os.environ["OPENAI_API_KEY"] = api_key
