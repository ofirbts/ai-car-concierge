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
from backend.database import init_db
from backend.orchestrator import ChatRequest, handle_chat
from backend.rag_service import PolicyRAGService

bootstrap()
init_db()
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
        "Scenario 1 — Family road-trip buyer",
        [
            "I'm looking for a car for my family",
            "four people, budget around 75000",
            "we do long family trips and cargo space matters",
            "need space for family trips",
            "what's the best value here?",
            "compare your top two",
            "reserve vehicle #55",
        ],
    ),
    (
        "Scenario 2 — City electric buyer",
        [
            "I mostly drive in the city and want something quiet",
            "it's just me and my partner",
            "budget around 75000",
            "I prefer electric or hybrid",
            "what would you personally shortlist first?",
            "hold your top pick for me",
        ],
    ),
    (
        "Scenario 3 — Budget-conscious practical buyer",
        [
            "I need a practical family car but I'm price-sensitive",
            "family of 4, mostly city and weekend drives",
            "budget is 60000 max",
            "these feel expensive, can we go cheaper?",
            "which one is the smartest value-for-money choice?",
            "reserve the best value one",
        ],
    ),
]


def main() -> None:
    for title, turns in SCENARIOS:
        run_conversation(title, turns)
    print(f"\n{'=' * 60}")
    print("Demo complete — 3 premium sales flows")
    print("=" * 60)


if __name__ == "__main__":
    main()
