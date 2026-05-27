from unittest.mock import patch

from backend.database import get_vehicle_by_id
from backend.intent import IntentKind
from backend.orchestrator import ChatResponse


def test_chat_returns_422_when_prices_ungrounded(api_client):
    vehicle = get_vehicle_by_id(16)
    assert vehicle is not None
    bad = ChatResponse(
        reply="Special today only at $999,999.",
        intent=IntentKind.INVENTORY_SEARCH,
        vehicles=[vehicle],
    )
    with patch("backend.main.handle_chat", return_value=bad):
        response = api_client.post("/api/chat", json={"message": "show me options"})
    assert response.status_code == 422
    data = response.json()
    assert data["validation_verdict"] == "REJECT"
    assert data.get("request_id")


def test_chat_returns_200_when_prices_grounded(api_client):
    vehicle = get_vehicle_by_id(16)
    assert vehicle is not None
    good = ChatResponse(
        reply=f"#{vehicle.id} is listed at ${vehicle.price:,.0f}.",
        intent=IntentKind.INVENTORY_SEARCH,
        vehicles=[vehicle],
    )
    with patch("backend.main.handle_chat", return_value=good):
        response = api_client.post("/api/chat", json={"message": "show me options"})
    assert response.status_code == 200
    assert response.json()["validation_verdict"] == "PASS"
