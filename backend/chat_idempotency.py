from __future__ import annotations

import re
import uuid

RESERVE_VEHICLE_RE = re.compile(
    r"\breserve\b.*?(?:vehicle\s*)?#?\s*(\d+)",
    re.IGNORECASE,
)


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
