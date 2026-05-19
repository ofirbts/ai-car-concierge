import contextvars
import uuid

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


def get_request_id() -> str:
    return _request_id.get()


def bind_request_id(value: str | None = None) -> str:
    rid = (value or "").strip() or str(uuid.uuid4())
    _request_id.set(rid)
    return rid
