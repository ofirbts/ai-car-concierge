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
    if ENV_FILE.is_file():
        key = (dotenv_values(ENV_FILE).get("API_KEY") or "").strip()
    return key


@pytest.fixture(scope="module")
def production_key():
    key = _production_api_key()
    if not key:
        pytest.skip("Set API_KEY in environment or .env for production smoke tests")
    return key


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
        json={"message": "Show me Tesla cars"},
        headers={"X-API-Key": production_key},
        timeout=90.0,
    )
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert data.get("intent") == "inventory_search"
