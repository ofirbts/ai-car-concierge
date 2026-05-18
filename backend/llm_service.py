from __future__ import annotations

from backend.gemini_service import generate_text

CONCIERGE_SYSTEM = (
    "You are Premium Auto Gallery's AI Car Concierge. "
    "Answer using ONLY the provided context. "
    "Never invent inventory, prices, or policy rules. "
    "If context says a pre-2022 vehicle is Pending De-listing, "
    "you must state it cannot be sold or reserved. "
    "Be concise, professional, and helpful."
)


def synthesize_reply(user_message: str, system_context: str) -> str | None:
    instruction = f"{CONCIERGE_SYSTEM}\n\nContext:\n{system_context}"
    return generate_text(instruction, user_message)
