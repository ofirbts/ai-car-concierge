from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from backend.database import get_job_record, request_job_cancel, save_job_record


class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobRecord(BaseModel):
    job_id: str
    state: JobState
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    cancel_requested: bool = False


class JobContext(BaseModel):
    job_id: str

    def __init__(self, job_id: str, cancel_event: threading.Event):
        super().__init__(job_id=job_id)
        self._cancel_event = cancel_event

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()


class JobBroker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._cancel_events: dict[str, threading.Event] = {}

    def submit(
        self,
        handler: Callable[[JobContext], dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        job_id = str(uuid.uuid4())
        cancel_event = threading.Event()
        record = JobRecord(
            job_id=job_id,
            state=JobState.PENDING,
            created_at=time.time(),
            metadata=metadata or {},
        )
        with self._lock:
            self._jobs[job_id] = record
            self._cancel_events[job_id] = cancel_event
            save_job_record(record.model_dump(mode="json"))

        thread = threading.Thread(
            target=self._run,
            args=(job_id, handler, cancel_event),
            daemon=True,
        )
        thread.start()
        return job_id

    def _run(
        self,
        job_id: str,
        handler: Callable[[JobContext], dict[str, Any]],
        cancel_event: threading.Event,
    ) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.state = JobState.RUNNING
            record.started_at = time.time()
            self._jobs[job_id] = record
            save_job_record(record.model_dump(mode="json"))

        context = JobContext(job_id=job_id, cancel_event=cancel_event)
        try:
            if cancel_event.is_set():
                self._mark_cancelled(job_id)
                return
            result = handler(context)
            if cancel_event.is_set():
                self._mark_cancelled(job_id)
                return
            with self._lock:
                record = self._jobs[job_id]
                record.state = JobState.COMPLETED
                record.finished_at = time.time()
                record.result = result
                self._jobs[job_id] = record
                save_job_record(record.model_dump(mode="json"))
        except Exception as exc:
            with self._lock:
                record = self._jobs[job_id]
                record.state = JobState.FAILED
                record.finished_at = time.time()
                record.error = str(exc)
                self._jobs[job_id] = record
                save_job_record(record.model_dump(mode="json"))

    def _mark_cancelled(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.state = JobState.CANCELLED
            record.finished_at = time.time()
            self._jobs[job_id] = record
            save_job_record(record.model_dump(mode="json"))

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            cancel_event = self._cancel_events.get(job_id)
            if cancel_event is None:
                if request_job_cancel(job_id):
                    db_record = get_job_record(job_id)
                    if db_record is not None:
                        self._jobs[job_id] = JobRecord.model_validate(db_record)
                    return True
                return False
            record = self._jobs[job_id]
            if record.state in {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}:
                return False
            cancel_event.set()
            record.cancel_requested = True
            save_job_record(record.model_dump(mode="json"))
            if record.state == JobState.PENDING:
                record.state = JobState.CANCELLED
                record.finished_at = time.time()
                self._jobs[job_id] = record
                save_job_record(record.model_dump(mode="json"))
            return True

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is not None:
                return record.model_copy(deep=True)
        db_record = get_job_record(job_id)
        if db_record is None:
            return None
        parsed = JobRecord.model_validate(db_record)
        with self._lock:
            self._jobs[job_id] = parsed
            self._cancel_events.setdefault(job_id, threading.Event())
        return parsed

