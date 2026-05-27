from fastapi.testclient import TestClient

from backend.config import reset_settings_cache


def test_experimental_routes_absent_when_disabled(isolated_db, monkeypatch):
    monkeypatch.setenv("ENABLE_EXPERIMENTAL", "false")
    reset_settings_cache()
    from backend.main import create_app

    with TestClient(create_app()) as client:
        openapi = client.get("/openapi.json").json()
        paths = openapi.get("paths", {})
        assert "/jobs/submit" not in paths
        assert "/skills/codebase-packager" not in paths
        assert "/governor/iteration/run" not in paths
        root = client.get("/").json()
        assert "experimental_api" not in root.get("features", {})
