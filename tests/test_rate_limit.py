from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded


def test_rate_limit_key_uses_api_key_when_present():
    from starlette.requests import Request
    from backend.main import rate_limit_key

    scope = {"type": "http", "headers": [(b"x-api-key", b"secret-abc")]}
    request = Request(scope)
    assert rate_limit_key(request) == "key:secret-abc"


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
    import uuid

    monkeypatch.setenv("CHAT_RATE_LIMIT", "1/second")
    from backend.config import reset_settings_cache
    from backend.main import limiter

    client_key = f"rate-limit-{uuid.uuid4()}"
    monkeypatch.setattr(limiter, "_key_func", lambda _request: client_key)
    reset_settings_cache()
    limiter.reset()
    first = api_client.post("/api/chat", json={"message": "Tesla inventory"})
    assert first.status_code == 200
    second = api_client.post("/api/chat", json={"message": "BMW inventory"})
    assert second.status_code == 429
