import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from backend.request_context import bind_request_id, get_request_id

logger = logging.getLogger("concierge.access")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = bind_request_id(request.headers.get("X-Request-ID"))
        start = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = get_request_id() or rid
        logger.info(
            "http_request request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
            rid,
            request.method,
            request.url.path,
            response.status_code,
            ms,
        )
        return response
