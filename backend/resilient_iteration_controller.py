from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from backend.idempotency_utils import stable_idempotency_key


StateType = dict[str, Any]
StepHandler = Callable[[StateType], StateType]


class JournalEvent(BaseModel):
    run_id: str
    event_type: str
    step_name: str | None = None
    timestamp: float
    payload: dict[str, Any] = Field(default_factory=dict)


class ControllerRunResult(BaseModel):
    run_id: str
    state: StateType
    completed_steps: list[str]
    last_event: str


class ResilientIterationController:
    def __init__(self, journal_path: str | Path):
        self._journal_path = Path(journal_path)

    def _append_event(self, event: JournalEvent) -> None:
        self._journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self._journal_path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json())
            handle.write("\n")

    def _load_events(self, run_id: str) -> list[JournalEvent]:
        events: list[JournalEvent] = []
        if not self._journal_path.exists():
            return events
        with self._journal_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                event = JournalEvent.model_validate_json(line)
                if event.run_id == run_id:
                    events.append(event)
        return events

    def replay(self, run_id: str, initial_state: StateType) -> tuple[StateType, list[str]]:
        state = dict(initial_state)
        completed: list[str] = []
        events = self._load_events(run_id)
        for event in events:
            if event.event_type == "STEP_COMPLETED" and event.step_name:
                state.update(event.payload.get("state_patch", {}))
                completed.append(event.step_name)
        return state, completed

    def run(
        self,
        run_id: str,
        initial_state: StateType,
        steps: list[tuple[str, StepHandler]],
    ) -> ControllerRunResult:
        state, completed = self.replay(run_id, initial_state)
        completed_set = set(completed)

        for step_name, handler in steps:
            if step_name in completed_set:
                continue
            self._append_event(
                JournalEvent(
                    run_id=run_id,
                    event_type="STEP_STARTED",
                    step_name=step_name,
                    timestamp=time.time(),
                )
            )
            try:
                next_state = handler(dict(state))
                patch_key = stable_idempotency_key(
                    "iteration_patch",
                    {"run_id": run_id, "step": step_name, "state": next_state},
                )
                patch = {"state_patch": next_state, "patch_key": patch_key}
                self._append_event(
                    JournalEvent(
                        run_id=run_id,
                        event_type="STEP_COMPLETED",
                        step_name=step_name,
                        timestamp=time.time(),
                        payload=patch,
                    )
                )
                state.update(next_state)
                completed.append(step_name)
                completed_set.add(step_name)
            except Exception as exc:
                self._append_event(
                    JournalEvent(
                        run_id=run_id,
                        event_type="STEP_FAILED",
                        step_name=step_name,
                        timestamp=time.time(),
                        payload={"error": str(exc)},
                    )
                )
                return ControllerRunResult(
                    run_id=run_id,
                    state=state,
                    completed_steps=completed,
                    last_event="STEP_FAILED",
                )

        self._append_event(
            JournalEvent(
                run_id=run_id,
                event_type="RUN_COMPLETED",
                timestamp=time.time(),
            )
        )
        return ControllerRunResult(
            run_id=run_id,
            state=state,
            completed_steps=completed,
            last_event="RUN_COMPLETED",
        )

