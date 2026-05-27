#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os

os.environ["GOOGLE_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""

from backend.config import bootstrap
from backend.conversation_state import DialoguePhase
from backend.database import init_db
from backend.orchestrator import ChatRequest, handle_chat
from backend.rag_service import PolicyRAGService

bootstrap()
init_db()
from backend.rag_service import PolicyRAGService

rag = PolicyRAGService(use_embeddings=False)


def run_conversation(title: str, turns: list[str]) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)
    session_id = None
    for idx, message in enumerate(turns, start=1):
        print(f"\nUser [{idx}]: {message}")
        response = handle_chat(
            ChatRequest(message=message, session_id=session_id, user_email="demo@example.com"),
            rag=rag,
        )
        session_id = response.session_id
        meta = {
            "intent": response.intent.value,
            "phase": response.dialogue_phase.value if response.dialogue_phase else None,
            "vehicles": [v.id for v in response.vehicles],
            "progress": response.conversation_progress,
        }
        print(f"Assistant: {response.reply}")
        print(f"[meta] {json.dumps(meta, ensure_ascii=False)}")


SCENARIOS = [
    (
        "Scenario 1 — Family with budget",
        [
            "I'm looking for a car for my family",
            "four people — two adults and two kids",
            "budget around 75000",
            "family trips, space is more important than fuel",
            "what's the best value?",
            "reserve vehicle #16",
        ],
    ),
    (
        "Scenario 2 — First-time buyer unsure",
        [
            "I'm not sure what car I want, first time buying",
            "mostly city driving, couple with a baby",
            "maybe 65000 budget",
            "recommend something",
        ],
    ),
    (
        "Scenario 3 — Compare and decide",
        [
            "help me find an SUV under 80000",
            "family of 4",
            "80000 max",
            "highway and city mix",
            "compare #17 and #36",
            "which is more affordable?",
        ],
    ),
    (
        "Scenario 4 — Purchase escalation",
        [
            "looking for a Tesla for daily commute under 70000",
            "just me and my partner",
            "70000",
            "city driving",
            "I want to buy the best match",
        ],
    ),
    (
        "Scenario 5 — Natural language discovery",
        [
            "something good for a family, not too expensive",
            "five people total",
            "60000",
            "kids car seats, need room",
            "show me options",
        ],
    ),
]


def main() -> None:
    for title, turns in SCENARIOS:
        run_conversation(title, turns)
    print(f"\n{'=' * 60}")
    print("Demo complete — 5 conversational purchase flows")
    print("=" * 60)


if __name__ == "__main__":
    main()
