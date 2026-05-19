from __future__ import annotations

import logging
import re
from enum import Enum

from pydantic import BaseModel, EmailStr

from backend.config import get_settings
from backend.database import SALES_MIN_YEAR
from backend.gemini_service import generate_structured

logger = logging.getLogger(__name__)

EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
VEHICLE_ID_RE = re.compile(r"(?:vehicle\s*)?(?:#|id\s*)(\d+)", re.IGNORECASE)
YEAR_IN_MESSAGE_RE = re.compile(r"\b(20\d{2})\b")

POLICY_KEYWORDS = (
    "refund",
    "test drive",
    "shipping",
    "delivery",
    "maintenance",
    "warranty",
    "policy",
    "service schedule",
    "contact support",
)
INVENTORY_KEYWORDS = (
    "car",
    "cars",
    "vehicle",
    "inventory",
    "stock",
    "find",
    "show",
    "suv",
    "sedan",
    "price",
    "cost",
    "under $",
    "below $",
    "tesla",
    "bmw",
    "audi",
)
PURCHASE_KEYWORDS = ("buy", "purchase", "order", "pay")
RESERVE_KEYWORDS = ("reserve", "hold", "book")

INTENT_SYSTEM = (
    "Classify dealership concierge messages. "
    "Use hybrid_rag when the user asks about BOTH inventory/pricing AND "
    "policies (refund, shipping, test drive, etc.) in one message. "
    "Use legacy_year_conflict when they ask specifically for model years "
    "2019, 2020, or 2021. Extract make, model, year, vehicle_id, user_email when present."
)


class IntentKind(str, Enum):
    INVENTORY_SEARCH = "inventory_search"
    POLICY_QUESTION = "policy_question"
    HYBRID_RAG = "hybrid_rag"
    PURCHASE_INTENT = "purchase_intent"
    RESERVE_INTENT = "reserve_intent"
    GENERAL_CHAT = "general_chat"
    LEGACY_YEAR_CONFLICT = "legacy_year_conflict"


class ExtractedIntent(BaseModel):
    intent: IntentKind
    make: str | None = None
    model: str | None = None
    year: int | None = None
    year_min: int | None = None
    price_max: float | None = None
    vehicle_id: int | None = None
    user_email: EmailStr | None = None


def extract_email(message: str, override: str | None) -> str | None:
    if override:
        return override
    match = EMAIL_RE.search(message)
    return match.group(0) if match else None


def extract_vehicle_id(message: str) -> int | None:
    match = VEHICLE_ID_RE.search(message)
    return int(match.group(1)) if match else None


def extract_year(message: str) -> int | None:
    match = re.search(r"\b(20\d{2})\b", message)
    return int(match.group(1)) if match else None


def extract_price_max(message: str) -> float | None:
    match = re.search(
        r"(?:under|below|max|less than)\s*\$?\s*([\d,]+)|\$?\s*([\d,]+)\s*(?:or less|max)",
        message.lower(),
    )
    if match:
        value = match.group(1) or match.group(2)
        return float(value.replace(",", "")) if value else None
    return None


def _inventory_makes() -> list[str]:
    from backend.database import list_distinct_makes

    return list_distinct_makes()


def extract_make_model(message: str) -> tuple[str | None, str | None]:
    lower = message.lower()
    found_make: str | None = None
    for make in _inventory_makes():
        if make.lower() in lower:
            found_make = make
            break
    model = None
    if found_make:
        patterns = [
            r"model\s+([a-z0-9][\w\s-]*)",
            r"(model\s*y|model\s*3|x5|q7|a4|gle|c-class|f-pace|range rover)",
        ]
        for pattern in patterns:
            m = re.search(pattern, lower)
            if m:
                model = m.group(1).strip().title()
                break
    return found_make, model


def has_policy_signal(message: str) -> bool:
    lower = message.lower()
    return any(k in lower for k in POLICY_KEYWORDS)


def has_inventory_signal(
    message: str, make: str | None, year: int | None, price_max: float | None
) -> bool:
    lower = message.lower()
    return bool(
        any(k in lower for k in INVENTORY_KEYWORDS) or make or year is not None or price_max
    )


def message_mentions_pre_2022_year(message: str) -> bool:
    for match in YEAR_IN_MESSAGE_RE.finditer(message):
        if int(match.group(1)) < SALES_MIN_YEAR:
            return True
    return False


def is_legacy_year_focus(extracted: ExtractedIntent, message: str) -> bool:
    if extracted.year is not None and extracted.year < SALES_MIN_YEAR:
        return True
    if extracted.year_min is not None and extracted.year_min < SALES_MIN_YEAR:
        return True
    return message_mentions_pre_2022_year(message)


def apply_legacy_year_override(extracted: ExtractedIntent, message: str) -> ExtractedIntent:
    if extracted.intent in (
        IntentKind.INVENTORY_SEARCH,
        IntentKind.HYBRID_RAG,
    ) and is_legacy_year_focus(extracted, message):
        extracted.intent = IntentKind.LEGACY_YEAR_CONFLICT
    return extracted


def classify_intent_rule_based(message: str, user_email: str | None = None) -> ExtractedIntent:
    lower = message.lower()
    email = extract_email(message, user_email)
    vehicle_id = extract_vehicle_id(message)
    year = extract_year(message)
    make, model = extract_make_model(message)
    price_max = extract_price_max(message)

    if any(k in lower for k in RESERVE_KEYWORDS):
        return ExtractedIntent(
            intent=IntentKind.RESERVE_INTENT,
            vehicle_id=vehicle_id,
            user_email=email,
            make=make,
            model=model,
            year=year,
        )
    if any(k in lower for k in PURCHASE_KEYWORDS):
        return ExtractedIntent(
            intent=IntentKind.PURCHASE_INTENT,
            vehicle_id=vehicle_id,
            user_email=email,
            make=make,
            model=model,
            year=year,
        )

    has_policy = has_policy_signal(message)
    has_inventory = has_inventory_signal(message, make, year, price_max)
    if has_policy and has_inventory:
        return apply_legacy_year_override(
            ExtractedIntent(
                intent=IntentKind.HYBRID_RAG,
                make=make,
                model=model,
                year=year,
                year_min=year,
                price_max=price_max,
            ),
            message,
        )
    if has_policy:
        return ExtractedIntent(intent=IntentKind.POLICY_QUESTION)
    if has_inventory:
        base = ExtractedIntent(
            intent=IntentKind.INVENTORY_SEARCH,
            make=make,
            model=model,
            year=year,
            year_min=year,
            price_max=price_max,
        )
        return apply_legacy_year_override(base, message)
    return ExtractedIntent(intent=IntentKind.GENERAL_CHAT)


def classify_intent(message: str, user_email: str | None = None) -> ExtractedIntent:
    if not get_settings().has_google_api():
        return classify_intent_rule_based(message, user_email)

    parsed = generate_structured(INTENT_SYSTEM, message, ExtractedIntent)
    if parsed is not None:
        if user_email and not parsed.user_email:
            parsed.user_email = user_email
        return apply_legacy_year_override(parsed, message)

    logger.warning("Gemini intent classification unavailable, using rules")
    return classify_intent_rule_based(message, user_email)
