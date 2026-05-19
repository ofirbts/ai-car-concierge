from __future__ import annotations

import logging
from typing import TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from backend.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_client: genai.Client | None = None


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
    try:
        response = client.models.generate_content(
            model=chat_model_name(),
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
            ),
        )
        text = response.text
        return text.strip() if text else None
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
    try:
        response = client.models.generate_content(
            model=chat_model_name(),
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.1,
            ),
        )
        text = response.text
        if not text:
            return None
        return schema.model_validate_json(text)
    except Exception as exc:
        logger.warning("Gemini structured output failed (%s): %s", type(exc).__name__, exc)
        return None


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = get_gemini_client()
    if client is None or not texts:
        return []
    settings = get_settings()
    try:
        response = client.models.embed_content(
            model=settings.gemini_embedding_model,
            contents=texts,
        )
        return [list(item.values) for item in response.embeddings]
    except Exception as exc:
        logger.warning("Gemini embed_texts failed (%s): %s", type(exc).__name__, exc)
        return []


def embed_query(query: str) -> list[float]:
    vectors = embed_texts([query])
    return vectors[0] if vectors else []
