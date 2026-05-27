from __future__ import annotations

import logging
from typing import TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from backend.config import get_settings
from backend.idempotency_utils import DedupStore, stable_idempotency_key
from backend.reliability import RetryPolicy, run_with_fallbacks

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_client: genai.Client | None = None
_dedup = DedupStore()


def get_gemini_client() -> genai.Client | None:
    global _client
    settings = get_settings()
    if not settings.has_google_api():
        return None
    if _client is None:
        _client = genai.Client(api_key=settings.google_api_key)
    return _client


def reset_gemini_client() -> None:
    global _client
    _client = None


def chat_model_name() -> str:
    settings = get_settings()
    if settings.use_quality_llm:
        return settings.gemini_chat_model_quality
    return settings.gemini_chat_model


def generate_text(system_instruction: str, user_message: str) -> str | None:
    client = get_gemini_client()
    if client is None:
        return None
    cache_key = stable_idempotency_key(
        "gemini_text",
        {
            "model": chat_model_name(),
            "system_instruction": system_instruction,
            "user_message": user_message,
        },
    )
    cached = _dedup.get(cache_key)
    if cached is not None:
        return cached
    try:
        policy = RetryPolicy(attempts=2, timeout_seconds=10.0, backoff_seconds=0.25)
        response = run_with_fallbacks(
            "gemini_generate_text",
            [
                lambda: client.models.generate_content(
                    model=chat_model_name(),
                    contents=user_message,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.3,
                    ),
                ),
                lambda: client.models.generate_content(
                    model=chat_model_name(),
                    contents=user_message,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.0,
                    ),
                ),
            ],
            policy,
        )
        text = response.text
        output = text.strip() if text else None
        if output is not None:
            _dedup.put(cache_key, output, ttl_seconds=600)
        return output
    except Exception as exc:
        logger.warning("Gemini generate_text failed (%s): %s", type(exc).__name__, exc)
        return None


def generate_structured(
    system_instruction: str,
    user_message: str,
    schema: type[T],
) -> T | None:
    client = get_gemini_client()
    if client is None:
        return None
    cache_key = stable_idempotency_key(
        "gemini_structured",
        {
            "model": chat_model_name(),
            "system_instruction": system_instruction,
            "user_message": user_message,
            "schema": schema.__name__,
        },
    )
    cached = _dedup.get(cache_key)
    if cached is not None:
        return schema.model_validate(cached)
    try:
        policy = RetryPolicy(attempts=2, timeout_seconds=10.0, backoff_seconds=0.25)
        response = run_with_fallbacks(
            "gemini_generate_structured",
            [
                lambda: client.models.generate_content(
                    model=chat_model_name(),
                    contents=user_message,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0.1,
                    ),
                ),
                lambda: client.models.generate_content(
                    model=chat_model_name(),
                    contents=user_message,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0.0,
                    ),
                ),
            ],
            policy,
        )
        text = response.text
        if not text:
            return None
        parsed = schema.model_validate_json(text)
        _dedup.put(cache_key, parsed.model_dump(mode="json"), ttl_seconds=600)
        return parsed
    except Exception as exc:
        logger.warning("Gemini structured output failed (%s): %s", type(exc).__name__, exc)
        return None


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = get_gemini_client()
    if client is None or not texts:
        return []
    cache_key = stable_idempotency_key(
        "gemini_embed_texts",
        {"model": get_settings().gemini_embedding_model, "texts": texts},
    )
    cached = _dedup.get(cache_key)
    if cached is not None:
        return cached
    settings = get_settings()
    try:
        policy = RetryPolicy(attempts=2, timeout_seconds=12.0, backoff_seconds=0.25)
        response = run_with_fallbacks(
            "gemini_embed_texts",
            [
                lambda: client.models.embed_content(
                    model=settings.gemini_embedding_model,
                    contents=texts,
                )
            ],
            policy,
        )
        vectors = [list(item.values) for item in response.embeddings]
        _dedup.put(cache_key, vectors, ttl_seconds=1200)
        return vectors
    except Exception as exc:
        logger.warning("Gemini embed_texts failed (%s): %s", type(exc).__name__, exc)
        return []


def embed_query(query: str) -> list[float]:
    vectors = embed_texts([query])
    return vectors[0] if vectors else []
