import os

import httpx
import pytest

PRODUCTION_URL = os.environ.get(
    "PRODUCTION_URL",
    "https://ai-car-concierge-a073.onrender.com",
).rstrip("/")


@pytest.mark.skipif(
    not os.environ.get("API_KEY", "").strip(),
    reason="Set API_KEY to run production smoke test",
)
def test_production_ready():
    response = httpx.get(f"{PRODUCTION_URL}/ready", timeout=60.0)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["vehicles"] == 100


@pytest.mark.skipif(
    not os.environ.get("API_KEY", "").strip(),
    reason="Set API_KEY to run production smoke test",
)
def test_production_chat_with_api_key():
    response = httpx.post(
        f"{PRODUCTION_URL}/api/chat",
        json={"message": "Show me Tesla cars"},
        headers={"X-API-Key": os.environ["API_KEY"].strip()},
        timeout=90.0,
    )
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert data.get("intent") == "inventory_search"
