from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_idempotency_key(namespace: str, payload: Any) -> str:
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


@dataclass(frozen=True)
class DedupEntry:
    value: Any
    expires_at: float


class DedupStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, DedupEntry] = {}

    def get(self, key: str) -> Any | None:
        now = time.time()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                self._entries.pop(key, None)
                return None
            return entry.value

    def put(self, key: str, value: Any, ttl_seconds: int) -> None:
        expires_at = time.time() + max(1, ttl_seconds)
        with self._lock:
            self._entries[key] = DedupEntry(value=value, expires_at=expires_at)

