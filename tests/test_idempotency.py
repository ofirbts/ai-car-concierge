import pytest

from backend.database import IdempotencyConflictError, reserve_vehicle


def test_reserve_idempotency_replays_same_vehicle(isolated_db):
    first = reserve_vehicle(16, idempotency_key="key-abc")
    second = reserve_vehicle(16, idempotency_key="key-abc")
    assert first.id == second.id
    assert second.stock_count == first.stock_count


def test_reserve_idempotency_conflict_different_vehicle(isolated_db):
    reserve_vehicle(16, idempotency_key="key-xyz")
    with pytest.raises(IdempotencyConflictError):
        reserve_vehicle(17, idempotency_key="key-xyz")


def test_api_reserve_idempotency_header(api_client):
    headers = {"Idempotency-Key": "api-key-1"}
    first = api_client.post("/vehicles/16/reserve", headers=headers)
    second = api_client.post("/vehicles/16/reserve", headers=headers)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["vehicle"]["stock_count"] == second.json()["vehicle"]["stock_count"]


def test_api_reserve_idempotency_conflict(api_client):
    api_client.post("/vehicles/16/reserve", headers={"Idempotency-Key": "conflict-key"})
    response = api_client.post("/vehicles/17/reserve", headers={"Idempotency-Key": "conflict-key"})
    assert response.status_code == 409


def test_chat_reserve_idempotency_same_key(api_client):
    payload = {"message": "reserve vehicle #16", "idempotency_key": "chat-reserve-16"}
    before = api_client.get("/vehicles/16").json()["stock_count"]
    first = api_client.post("/api/chat", json=payload)
    second = api_client.post("/api/chat", json=payload)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["reserved_vehicle"]["stock_count"] == second.json()["reserved_vehicle"]["stock_count"]
    after = api_client.get("/vehicles/16").json()["stock_count"]
    assert after == before - 1
