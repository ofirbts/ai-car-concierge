import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app, headers={"X-API-Key": ""})


def test_session_id_persists_across_turns(isolated_db: object) -> None:
    first = client.post("/api/chat", json={"message": "I need a family car, 4 people"})
    assert first.status_code == 200
    data1 = first.json()
    session_id = data1.get("session_id")
    assert session_id, "First turn must return a session_id"

    second = client.post("/api/chat", json={"message": "budget is 75000", "session_id": session_id})
    assert second.status_code == 200
    data2 = second.json()
    assert data2.get("session_id") == session_id, (
        "session_id must remain the same across turns"
    )

    assert data2.get("conversation_progress"), (
        "conversation_progress should be non-empty after two turns"
    )


def test_vehicle_cards_hidden_during_discovery(isolated_db: object) -> None:
    r = client.post("/api/chat", json={"message": "I'm looking for a car"})
    assert r.status_code == 200
    data = r.json()
    if not data.get("show_vehicle_cards"):
        assert data.get("vehicles") == [] or data.get("vehicles") is None, (
            "When show_vehicle_cards is False, vehicles list must be empty"
        )


def test_vehicle_cards_present_after_full_discovery(isolated_db: object) -> None:
    session_id = None

    steps = [
        "I want a family car, 4 people, family trips",
        "budget 75000",
    ]
    last_response = None
    for msg in steps:
        r = client.post("/api/chat", json={"message": msg, "session_id": session_id})
        assert r.status_code == 200
        data = r.json()
        session_id = data.get("session_id")
        last_response = data

    assert last_response is not None
    assert session_id is not None


def test_search_explanation_present_on_recommendations(isolated_db: object) -> None:
    session_id = None
    found_explanation = False

    for msg in [
        "family car, 4 people, family trips",
        "budget 75000",
        "show me options",
    ]:
        r = client.post("/api/chat", json={"message": msg, "session_id": session_id})
        assert r.status_code == 200
        data = r.json()
        session_id = data.get("session_id")
        if data.get("search_explanation"):
            found_explanation = True
            explanation = data["search_explanation"]
            assert "applied_filters" in explanation, "search_explanation must have applied_filters"
            assert "excluded" in explanation, "search_explanation must have excluded list"

    assert found_explanation, (
        "At least one turn with recommendations must include search_explanation"
    )
