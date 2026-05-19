from __future__ import annotations

import re
import uuid

RESERVE_VEHICLE_RE = re.compile(
    r"\breserve\b.*?(?:vehicle\s*)?#?\s*(\d+)",
    re.IGNORECASE,
)
PURCHASE_RE = re.compile(r"\b(?:buy|purchase|order)\b", re.I)
VEHICLE_ID_RE = re.compile(r"(?:vehicle\s*)?#?\s*(\d+)", re.I)


def reserve_vehicle_id_from_message(message: str) -> int | None:
    match = RESERVE_VEHICLE_RE.search(message)
    return int(match.group(1)) if match else None


def stable_reserve_idempotency_key(
    message: str,
    known_keys: dict[str, str],
) -> str | None:
    vehicle_id = reserve_vehicle_id_from_message(message)
    if vehicle_id is None:
        return None
    key_id = str(vehicle_id)
    if key_id not in known_keys:
        known_keys[key_id] = str(uuid.uuid4())
    return known_keys[key_id]


def stable_purchase_idempotency_key(
    message: str,
    user_email: str | None,
    known_keys: dict[str, str],
) -> str | None:
    if not PURCHASE_RE.search(message) or not user_email:
        return None
    vid = VEHICLE_ID_RE.search(message)
    slot = f"purchase:{vid.group(1) if vid else 'inquiry'}:{user_email.strip().lower()}"
    if slot not in known_keys:
        known_keys[slot] = str(uuid.uuid4())
    return known_keys[slot]
