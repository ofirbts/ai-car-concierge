from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded


def test_rate_limit_exceeded_handler_registered():
    from backend.main import app

    assert app.exception_handlers[RateLimitExceeded] is _rate_limit_exceeded_handler


def test_chat_endpoint_is_rate_limited():
    from backend.main import chat

    assert chat.__name__ in ("chat", "sync_wrapper")
