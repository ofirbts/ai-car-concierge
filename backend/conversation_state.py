from __future__ import annotations

import json
import sqlite3
import uuid
from enum import Enum

from pydantic import BaseModel, EmailStr, Field

from backend.database import ensure_app_tables, get_connection


class DialoguePhase(str, Enum):
    DISCOVERY = "discovery"
    RECOMMENDING = "recommending"
    COMPARING = "comparing"
    RESERVE = "reserve"
    PURCHASE = "purchase"
    COMPLETED = "completed"


class ConversationState(BaseModel):
    session_id: str
    budget: float | None = None
    body_type: str | None = None
    fuel_preference: str | None = None
    passengers: int | None = None
    family_size: int | None = None
    use_case: str | None = None
    must_have_features: list[str] = Field(default_factory=list)
    timeline: str | None = None
    shortlist_ids: list[int] = Field(default_factory=list)
    contact_email: EmailStr | None = None
    space_priority: str | None = None
    phase: DialoguePhase = DialoguePhase.DISCOVERY
    turn_count: int = 0
    last_recommended_ids: list[int] = Field(default_factory=list)
    compare_vehicle_ids: list[int] = Field(default_factory=list)

    def filled_slots(self) -> dict[str, object]:
        out: dict[str, object] = {}
        if self.budget is not None:
            out["budget"] = self.budget
        if self.body_type:
            out["body_type"] = self.body_type
        if self.fuel_preference:
            out["fuel_preference"] = self.fuel_preference
        if self.passengers is not None:
            out["passengers"] = self.passengers
        if self.family_size is not None:
            out["family_size"] = self.family_size
        if self.use_case:
            out["use_case"] = self.use_case
        if self.must_have_features:
            out["must_have_features"] = self.must_have_features
        if self.timeline:
            out["timeline"] = self.timeline
        if self.shortlist_ids:
            out["shortlist_ids"] = self.shortlist_ids
        if self.contact_email:
            out["contact_email"] = str(self.contact_email)
        if self.space_priority:
            out["space_priority"] = self.space_priority
        return out

    def has_discovery_basics(self) -> bool:
        passenger_info = self.passengers is not None or self.family_size is not None
        budget_info = self.budget is not None
        preference_info = bool(self.use_case or self.body_type or self.space_priority)
        return passenger_info and budget_info and preference_info

    def bump_turn(self) -> None:
        self.turn_count += 1


CONVERSATION_SESSIONS_DDL = """
CREATE TABLE IF NOT EXISTS conversation_sessions (
    session_id TEXT PRIMARY KEY,
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def ensure_conversation_tables() -> None:
    ensure_app_tables()
    conn = get_connection()
    try:
        conn.executescript(CONVERSATION_SESSIONS_DDL)
        conn.commit()
    finally:
        conn.close()


def new_session_id() -> str:
    return str(uuid.uuid4())


def load_conversation_state(session_id: str | None) -> ConversationState | None:
    if not session_id:
        return None
    ensure_conversation_tables()
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT state_json FROM conversation_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return ConversationState.model_validate(json.loads(row["state_json"]))


def save_conversation_state(state: ConversationState) -> None:
    ensure_conversation_tables()
    payload = state.model_dump(mode="json")
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO conversation_sessions (session_id, state_json, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(session_id) DO UPDATE SET
                state_json = excluded.state_json,
                updated_at = datetime('now')
            """,
            (state.session_id, json.dumps(payload, ensure_ascii=False)),
        )
        conn.commit()
    finally:
        conn.close()


def get_or_create_state(session_id: str | None) -> ConversationState:
    if session_id:
        existing = load_conversation_state(session_id)
        if existing is not None:
            return existing
        return ConversationState(session_id=session_id)
    return ConversationState(session_id=new_session_id())
