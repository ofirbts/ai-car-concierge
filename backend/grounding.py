from __future__ import annotations

import re
from typing import Sequence

from backend.database import Vehicle


def prices_mentioned_in_reply(reply: str) -> set[int]:
    return {int(amount.replace(",", "")) for amount in re.findall(r"\$([\d,]+)", reply)}


def allowed_prices_for_vehicles(
    vehicles: Sequence[Vehicle],
    *,
    reserved_vehicle: Vehicle | None = None,
) -> set[int]:
    allowed = {int(vehicle.price) for vehicle in vehicles}
    if reserved_vehicle is not None:
        allowed.add(int(reserved_vehicle.price))
    return allowed


def reply_prices_grounded(
    reply: str,
    vehicles: Sequence[Vehicle],
    *,
    reserved_vehicle: Vehicle | None = None,
) -> bool:
    mentioned = prices_mentioned_in_reply(reply)
    if not mentioned:
        return True
    allowed = allowed_prices_for_vehicles(vehicles, reserved_vehicle=reserved_vehicle)
    if not allowed:
        return False
    return mentioned <= allowed
