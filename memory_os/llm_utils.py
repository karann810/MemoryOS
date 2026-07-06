"""LLM and embedding adapters for memory_os.

Supports user-provided LLM objects with a generic interface.
If no user-provided object is given, falls back to OpenAI.
"""

import os
from typing import Any, Optional
from openai import OpenAI


def _extract_choice_content(choice: Any) -> str:
    if choice is None:
        return ""
    if hasattr(choice, "message"):
        message = choice.message
        if hasattr(message, "content"):
            return str(message.content).strip()
    if isinstance(choice, dict):
        message = choice.get("message")
        if isinstance(message, dict):
            return str(message.get("content", "")).strip()
        if isinstance(choice.get("text"), str):
            return choice["text"].strip()
    if hasattr(choice, "content"):
        return str(choice.content).strip()
    return str(choice).strip()


def parse_llm_response(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()

    if isinstance(response, dict):
        if "choices" in response and response["choices"]:
            return _extract_choice_content(response["choices"][0])
        if "output_text" in response:
            return str(response["output_text"]).strip()
        if "text" in response:
            return str(response["text"]).strip()

    if hasattr(response, "choices"):
        choices = response.choices
        if choices:
            return _extract_choice_content(choices[0])

    if hasattr(response, "output_text"):
        return str(response.output_text).strip()

    if hasattr(response, "text"):
        return str(response.text).strip()

    return str(response).strip()


def invoke_llm(
    llm: Any,
    messages: list[dict],
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 500,
    openai_key: Optional[str] = None,
    **kwargs,
) -> str:
    """Invoke a user-provided LLM or fallback OpenAI client."""
    if llm is None:
        llm = OpenAI(api_key=openai_key or os.getenv("OPENAI_API_KEY"))

    if hasattr(llm, "invoke"):
        response = llm.invoke(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return parse_llm_response(response)

    if hasattr(llm, "chat") and hasattr(llm.chat, "completions"):
        response = llm.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return parse_llm_response(response)

    if hasattr(llm, "chat") and hasattr(llm.chat, "complete"):
        response = llm.chat.complete(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return parse_llm_response(response)

    raise AttributeError("Provided llm object does not support invoke or chat completion.")


def embed_text(
    embedder: Any,
    text: str,
    model: Optional[str] = None,
    openai_key: Optional[str] = None,
) -> list[float]:
    """Embed text with a user-provided embedder or fallback OpenAI."""
    if embedder is not None:
        if hasattr(embedder, "embed"):
            vector = embedder.embed(text)
            if isinstance(vector, list) and vector and isinstance(vector[0], list):
                return vector[0]
            return vector

        if hasattr(embedder, "embeddings") and hasattr(embedder.embeddings, "create"):
            response = embedder.embeddings.create(model=model, input=text)
            return response.data[0].embedding

        if hasattr(embedder, "embeddings") and hasattr(embedder.embeddings, "embed"):
            vector = embedder.embeddings.embed(text)
            if isinstance(vector, list) and vector and isinstance(vector[0], list):
                return vector[0]
            return vector

        raise AttributeError("Provided embedder does not support embed or embeddings.create.")

    api_model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    client = OpenAI(api_key=openai_key or os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(model=api_model, input=text)
    return response.data[0].embedding
