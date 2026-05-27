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
    assert root_data.get("version") in ("1.1.0", "1.2.0"), root_data
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
