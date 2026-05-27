from unittest.mock import patch

from backend.version import APP_VERSION


def test_root_lists_entrypoints(api_client):
    response = api_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["docs"] == "/docs"
    assert data["version"] == APP_VERSION
    assert data["openapi"] == "/openapi.json"
    assert data["features"]["conversational_sales"] is True
    assert "chat" in data


def test_health_is_liveness_only(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "vehicles" not in response.json()


def test_ready_reports_inventory_and_policies(api_client):
    response = api_client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["vehicles"] == 100
    assert data["policy_chunks"] >= 10
    assert data["rag_mode"] in ("keyword", "gemini_embeddings")


def test_list_vehicles_filter_make(api_client):
    response = api_client.get("/vehicles", params={"make": "Tesla", "limit": 5})
    assert response.status_code == 200
    data = response.json()
    assert len(data) <= 5
    assert all("Tesla" in item["make"] for item in data)


def test_get_vehicle_by_id(api_client):
    response = api_client.get("/vehicles/16")
    assert response.status_code == 200
    assert response.json()["id"] == 16


def test_get_vehicle_not_found(api_client):
    response = api_client.get("/vehicles/99999")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert data.get("request_id") == response.headers.get("X-Request-ID")


def test_reserve_sellable_vehicle(api_client):
    before = api_client.get("/vehicles/16").json()["stock_count"]
    response = api_client.post("/vehicles/16/reserve")
    assert response.status_code == 200
    after = response.json()["vehicle"]["stock_count"]
    assert after == before - 1


def test_reserve_pre_2022_returns_409(api_client):
    response = api_client.post("/vehicles/5/reserve")
    assert response.status_code == 409
    assert "2022" in response.json()["error"]


def test_chat_policy_question(api_client):
    response = api_client.post(
        "/api/chat",
        json={"message": "What is your refund policy for deposits?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "policy_question"
    assert data["policy_context_used"] is True
    assert "refund" in data["reply"].lower()
    assert data.get("request_id")


def test_chat_response_includes_request_id(api_client):
    response = api_client.post(
        "/api/chat",
        json={"message": "Hello"},
    )
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID")
    assert response.json().get("request_id") == response.headers.get("X-Request-ID")


def test_chat_inventory(api_client):
    response = api_client.post(
        "/api/chat",
        json={"message": "Show me Tesla cars in inventory"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "inventory_search"
    assert len(data["vehicles"]) > 0


def test_chat_reserve_blocked_pre_2022_returns_409(api_client):
    response = api_client.post(
        "/api/chat",
        json={"message": "reserve vehicle #5"},
    )
    assert response.status_code == 409
    data = response.json()
    assert data["blocked"] is True
    assert data["reserved_vehicle"] is None


def test_chat_reserve_success(api_client):
    response = api_client.post(
        "/api/chat",
        json={"message": "reserve vehicle #16"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["reserved_vehicle"] is not None
    assert data["reserved_vehicle"]["id"] == 16


def test_chat_purchase_requires_email(api_client):
    response = api_client.post(
        "/api/chat",
        json={"message": "I want to buy vehicle #16"},
    )
    assert response.status_code == 200
    assert "email" in response.json()["reply"].lower()


def test_chat_purchase_blocked_pre_2022_returns_409(api_client):
    response = api_client.post(
        "/api/chat",
        json={
            "message": "I want to buy vehicle #5",
            "user_email": "buyer@example.com",
        },
    )
    assert response.status_code == 409
    data = response.json()
    assert data["blocked"] is True
    assert data["email_sent"] is False


def test_chat_purchase_inquiry_without_vehicle_sends_email(api_client):
    with patch("backend.orchestrator.send_purchase_inquiry_email") as mock_inquiry:
        from backend.automations import EmailResult

        mock_inquiry.return_value = EmailResult(sent=True)
        response = api_client.post(
            "/api/chat",
            json={
                "message": "I want to buy a family SUV",
                "user_email": "buyer@example.com",
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "purchase_intent"
    assert data["email_sent"] is True
    mock_inquiry.assert_called_once()


def test_policies_search_endpoint(api_client):
    response = api_client.get("/policies/search", params={"q": "test drive"})
    assert response.status_code == 200
    assert response.json()["chunks"]
