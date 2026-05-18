from __future__ import annotations

import logging

from openai import OpenAI

from backend.config import get_settings

logger = logging.getLogger(__name__)

CHAT_MODEL = "gpt-4o-mini"
CHAT_MODEL_QUALITY = "gpt-4o"


def get_openai_client() -> OpenAI | None:
    settings = get_settings()
    if not settings.openai_api_key.strip():
        return None
    return OpenAI(api_key=settings.openai_api_key)


def synthesize_reply(user_message: str, system_context: str) -> str | None:
    client = get_openai_client()
    if client is None:
        return None

    settings = get_settings()
    model = CHAT_MODEL_QUALITY if settings.use_quality_llm else CHAT_MODEL
    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Premium Auto Gallery's AI Car Concierge. "
                        "Answer using ONLY the provided context. "
                        "Never invent inventory, prices, or policy rules. "
                        "If context says a pre-2022 vehicle is Pending De-listing, "
                        "you must state it cannot be sold or reserved. "
                        "Be concise, professional, and helpful.\n\n"
                        f"Context:\n{system_context}"
                    ),
                },
                {"role": "user", "content": user_message},
            ],
        )
        text = completion.choices[0].message.content
        return text.strip() if text else None
    except Exception as exc:
        logger.warning("LLM reply synthesis failed: %s", exc)
        return None
