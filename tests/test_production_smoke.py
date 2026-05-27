import os
from pathlib import Path

import httpx
import pytest
from dotenv import dotenv_values

PRODUCTION_URL = os.environ.get(
    "PRODUCTION_URL",
    "https://ai-car-concierge-a073.onrender.com",
).rstrip("/")
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def _production_api_key() -> str:
    key = os.environ.get("API_KEY", "").strip()
    if key:
        return key
    if os.environ.get("REQUIRE_PRODUCTION_API_KEY") == "true":
        return ""
    if ENV_FILE.is_file():
        key = (dotenv_values(ENV_FILE).get("API_KEY") or "").strip()
    return key


def _require_production_key_in_ci() -> None:
    if os.environ.get("REQUIRE_PRODUCTION_API_KEY") == "true" and not _production_api_key():
        pytest.fail(
            "PRODUCTION_API_KEY GitHub secret is required for the smoke-prod CI job "
            "(Settings → Secrets → Actions)."
        )


@pytest.fixture(scope="module")
def production_key():
    _require_production_key_in_ci()
    key = _production_api_key()
    if not key:
        pytest.skip("Set API_KEY in environment or .env for production smoke tests")
    return key


def test_production_smoke_requires_api_key_secret_in_ci():
    _require_production_key_in_ci()


def test_production_ready():
    response = httpx.get(f"{PRODUCTION_URL}/ready", timeout=60.0)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["vehicles"] == 100


def test_production_chat_requires_key():
    response = httpx.post(
        f"{PRODUCTION_URL}/api/chat",
        json={"message": "Tesla"},
        timeout=60.0,
    )
    assert response.status_code == 401


def test_production_chat_with_api_key(production_key):
    response = httpx.post(
        f"{PRODUCTION_URL}/api/chat",
        json={"message": "Show me Tesla cars in inventory"},
        headers={"X-API-Key": production_key},
        timeout=90.0,
    )
    if response.status_code == 401:
        pytest.fail(
            "X-API-Key rejected by production API. "
            "Update GitHub secret PRODUCTION_API_KEY to match Render env API_KEY."
        )
    assert response.status_code == 200, response.text[:300]
    data = response.json()
    assert "reply" in data
    assert "tesla" in data["reply"].lower()
    assert len(data["reply"]) > 0


def test_production_api_version_and_conversational_sales(production_key):
    root = httpx.get(f"{PRODUCTION_URL}/", timeout=60.0)
    assert root.status_code == 200
    root_data = root.json()
    from backend.version import PRODUCTION_VERSION_ALLOWLIST

    assert root_data.get("version") in PRODUCTION_VERSION_ALLOWLIST, root_data
    assert root_data.get("features", {}).get("conversational_sales") is True

    response = httpx.post(
        f"{PRODUCTION_URL}/api/chat",
        json={"message": "I'm looking for a family car"},
        headers={"X-API-Key": production_key},
        timeout=90.0,
    )
    assert response.status_code == 200, response.text[:300]
    data = response.json()
    assert data.get("session_id"), data
    assert data.get("dialogue_phase") == "discovery", data
    assert "?" in data["reply"], data["reply"][:200]
    assert data.get("intent") == "general_chat"


def test_production_reserve_decrements_stock(production_key):
    vehicle_id = 16
    headers = {"X-API-Key": production_key}
    before_resp = httpx.get(
        f"{PRODUCTION_URL}/vehicles/{vehicle_id}",
        headers=headers,
        timeout=60.0,
    )
    assert before_resp.status_code == 200, before_resp.text[:200]
    before_stock = before_resp.json()["stock_count"]
    if before_stock < 1:
        pytest.skip(f"Vehicle #{vehicle_id} has no stock for reserve smoke test")

    import uuid

    reserve = httpx.post(
        f"{PRODUCTION_URL}/api/chat",
        json={
            "message": f"reserve vehicle #{vehicle_id}",
            "idempotency_key": f"smoke-reserve-{uuid.uuid4()}",
        },
        headers=headers,
        timeout=90.0,
    )
    assert reserve.status_code == 200, reserve.text[:300]
    reserve_data = reserve.json()
    assert reserve_data.get("reserved_vehicle"), reserve_data
    assert reserve_data["reserved_vehicle"]["id"] == vehicle_id

    after_resp = httpx.get(
        f"{PRODUCTION_URL}/vehicles/{vehicle_id}",
        headers=headers,
        timeout=60.0,
    )
    assert after_resp.status_code == 200
    after_stock = after_resp.json()["stock_count"]
    assert after_stock == before_stock - 1
