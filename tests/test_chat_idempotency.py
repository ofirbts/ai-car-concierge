from backend.chat_idempotency import (
    reserve_vehicle_id_from_message,
    stable_reserve_idempotency_key,
)


def test_reserve_vehicle_id_from_message():
    assert reserve_vehicle_id_from_message("reserve vehicle #16") == 16
    assert reserve_vehicle_id_from_message("Please reserve #42") == 42
    assert reserve_vehicle_id_from_message("Show me Tesla cars") is None


def test_stable_key_reused_for_same_vehicle():
    keys: dict[str, str] = {}
    first = stable_reserve_idempotency_key("reserve vehicle #16", keys)
    second = stable_reserve_idempotency_key("reserve vehicle #16 again", keys)
    assert first is not None
    assert first == second


def test_stable_key_differs_by_vehicle():
    keys: dict[str, str] = {}
    k16 = stable_reserve_idempotency_key("reserve vehicle #16", keys)
    k17 = stable_reserve_idempotency_key("reserve vehicle #17", keys)
    assert k16 != k17


def test_non_reserve_message_returns_none():
    keys: dict[str, str] = {}
    assert stable_reserve_idempotency_key("Tesla under 70000", keys) is None
