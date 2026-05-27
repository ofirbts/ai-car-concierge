from __future__ import annotations

import concurrent.futures
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


class ReliabilityError(RuntimeError):
    def __init__(self, operation: str, message: str) -> None:
        super().__init__(f"{operation}: {message}")
        self.operation = operation
        self.message = message


class TimeoutFailure(ReliabilityError):
    pass


class RetryExhausted(ReliabilityError):
    pass


class FallbackExhausted(ReliabilityError):
    pass


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 2
    timeout_seconds: float = 8.0
    backoff_seconds: float = 0.25


def run_with_retries(
    operation: str,
    call: Callable[[], T],
    policy: RetryPolicy,
) -> T:
    last_error: Exception | None = None
    for attempt in range(1, policy.attempts + 1):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(call)
                return future.result(timeout=policy.timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            last_error = TimeoutFailure(operation, f"timeout after {policy.timeout_seconds:.2f}s")
            if attempt < policy.attempts:
                time.sleep(policy.backoff_seconds * attempt)
            else:
                break
        except Exception as exc:
            last_error = exc
            if attempt < policy.attempts:
                time.sleep(policy.backoff_seconds * attempt)
            else:
                break
    if isinstance(last_error, ReliabilityError):
        raise RetryExhausted(operation, last_error.message) from last_error
    if last_error is None:
        raise RetryExhausted(operation, "unknown retry failure")
    raise RetryExhausted(operation, str(last_error)) from last_error


def run_with_fallbacks(
    operation: str,
    calls: list[Callable[[], T]],
    policy: RetryPolicy,
) -> T:
    if not calls:
        raise FallbackExhausted(operation, "no calls provided")
    last_error: Exception | None = None
    for call in calls:
        try:
            return run_with_retries(operation, call, policy)
        except Exception as exc:
            last_error = exc
    if isinstance(last_error, RetryExhausted):
        raise FallbackExhausted(operation, last_error.message) from last_error
    if last_error is None:
        raise FallbackExhausted(operation, "unknown fallback failure")
    raise FallbackExhausted(operation, str(last_error)) from last_error

