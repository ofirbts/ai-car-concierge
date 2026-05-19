from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client_with_key(isolated_db):
    with patch("backend.security.get_settings") as mock_settings:
        mock_settings.return_value.api_key = "secret-test-key"
        from backend.main import app

        with TestClient(app) as client:
            yield client


def test_api_key_not_required_when_unset(api_client):
    response = api_client.get("/vehicles", params={"limit": 1})
    assert response.status_code == 200


def test_api_key_rejects_missing(api_client_with_key):
    response = api_client_with_key.get("/vehicles", params={"limit": 1})
    assert response.status_code == 401


def test_api_key_accepts_valid_header(api_client_with_key):
    response = api_client_with_key.get(
        "/vehicles",
        params={"limit": 1},
        headers={"X-API-Key": "secret-test-key"},
    )
    assert response.status_code == 200
