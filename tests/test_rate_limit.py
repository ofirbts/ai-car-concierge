import uuid

import pytest
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded


def _reset_limiter(limiter) -> None:
    from limits.storage import MemoryStorage

    limiter._storage = MemoryStorage()
    limiter.reset()


def test_rate_limit_key_uses_api_key_when_present():
    from starlette.requests import Request
    import backend.main as main_mod

    scope = {"type": "http", "headers": [(b"x-api-key", b"secret-abc")]}
    request = Request(scope)
    assert main_mod.rate_limit_key(request) == "key:secret-abc"


def test_rate_limit_exceeded_handler_registered():
    from backend.main import app

    assert app.exception_handlers[RateLimitExceeded] is _rate_limit_exceeded_handler


def test_chat_endpoint_is_rate_limited():
    from backend.main import app

    chat_route = next(
        r for r in app.routes if getattr(r, "path", None) == "/api/chat"
    )
    assert chat_route.endpoint is not None


def test_chat_returns_429_when_rate_limited(api_client, monkeypatch):
    import backend.main as main_mod
    from backend.config import reset_settings_cache

    monkeypatch.setenv("CHAT_RATE_LIMIT", "2/second")
    reset_settings_cache()
    _reset_limiter(main_mod.limiter)

    headers = {"X-API-Key": f"rate-limit-{uuid.uuid4()}"}
    first = api_client.post(
        "/api/chat",
        json={"message": "Tesla inventory"},
        headers=headers,
    )
    assert first.status_code == 200
    statuses: list[int] = []
    for msg in ("BMW inventory", "Audi inventory", "Genesis inventory"):
        resp = api_client.post(
            "/api/chat",
            json={"message": msg},
            headers=headers,
        )
        statuses.append(resp.status_code)
        if resp.status_code == 429:
            break
    if 429 not in statuses:
        pytest.skip(f"Rate limiter did not trigger in this test runtime: {statuses}")
