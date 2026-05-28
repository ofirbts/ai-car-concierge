import json
import re
from pathlib import Path

import pytest

from backend.orchestrator import ChatRequest, ChatResponse, handle_chat
from backend.rag_service import PolicyRAGService

FIXTURES_PATH = Path(__file__).parent / "fixtures" / "conversation_golden.jsonl"
HEBREW_RE = re.compile(r"[\u0590-\u05FF]")


def _rag() -> PolicyRAGService:
    return PolicyRAGService(use_embeddings=False)


def _has_hebrew(text: str) -> bool:
    return bool(HEBREW_RE.search(text))


def _too_similar(a: str, b: str, threshold: float = 0.75) -> bool:
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return False
    overlap = words_a & words_b
    union = words_a | words_b
    return len(overlap) / len(union) > threshold


def _load_scenarios() -> list[dict]:
    if not FIXTURES_PATH.exists():
        return []
    scenarios = []
    for line in FIXTURES_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            scenarios.append(json.loads(line))
    return scenarios


SCENARIOS = _load_scenarios()


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["scenario_id"] for s in SCENARIOS])
def test_golden_scenario(scenario: dict, isolated_db: object) -> None:
    rag = _rag()
    session_id: str | None = None
    prev_reply: str | None = None

    for i, turn in enumerate(scenario["turns"]):
        user_msg: str = turn["user"]
        request = ChatRequest(message=user_msg, session_id=session_id)
        response: ChatResponse = handle_chat(request, rag=rag)
        session_id = response.session_id
        reply = response.reply

        assert reply and len(reply.strip()) > 5, (
            f"[{scenario['scenario_id']} turn {i}] reply is empty or too short: {reply!r}"
        )

        if turn.get("assert_language") == "he":
            assert _has_hebrew(reply), (
                f"[{scenario['scenario_id']} turn {i}] expected Hebrew in reply, got: {reply!r}"
            )

        if turn.get("assert_has_vehicles") is False:
            if response.show_vehicle_cards:
                assert response.vehicles == [], (
                    f"[{scenario['scenario_id']} turn {i}] expected no vehicles, got: {[v.id for v in response.vehicles]}"
                )

        contains_any: list[str] = turn.get("assert_contains_any", [])
        if contains_any:
            lower_reply = reply.lower()
            assert any(kw.lower() in lower_reply for kw in contains_any), (
                f"[{scenario['scenario_id']} turn {i}] none of {contains_any} found in: {reply!r}"
            )

        not_contains_all: list[str] = turn.get("assert_not_contains_all", [])
        if not_contains_all and len(not_contains_all) >= 2:
            lower_reply = reply.lower()
            all_present = all(kw.lower() in lower_reply for kw in not_contains_all)
            assert not all_present, (
                f"[{scenario['scenario_id']} turn {i}] all of {not_contains_all} appeared in reply — likely a copy-paste repeat: {reply!r}"
            )

        if turn.get("assert_not_too_similar_to_prev") and prev_reply is not None:
            assert not _too_similar(reply, prev_reply), (
                f"[{scenario['scenario_id']} turn {i}] reply too similar to previous turn.\n"
                f"  prev: {prev_reply!r}\n  curr: {reply!r}"
            )

        prev_reply = reply


def test_aura_like_transcript_no_loop(isolated_db: object) -> None:
    rag = _rag()
    msgs = [
        "I mostly drive in the city and want something quiet",
        "four people, budget 75000",
        "I'm price-sensitive and want the cheapest practical option",
        "budget is 60000 max",
        "תענה לי בעברית",
        "bad idea samshing else",
    ]
    replies: list[str] = []
    prev_vehicles: list[int] = []
    session_id: str | None = None
    consecutive_identical_recs = 0

    for msg in msgs:
        request = ChatRequest(message=msg, session_id=session_id)
        response: ChatResponse = handle_chat(request, rag=rag)
        session_id = response.session_id
        reply = response.reply

        assert reply and len(reply.strip()) > 5, f"Empty reply for: {msg!r}"

        if "עברית" in msg:
            assert _has_hebrew(reply), (
                f"Expected Hebrew reply after language switch, got: {reply!r}"
            )

        curr_vehicles = sorted([v.id for v in response.vehicles]) if response.vehicles else []
        if curr_vehicles and curr_vehicles == prev_vehicles:
            consecutive_identical_recs += 1
            assert consecutive_identical_recs < 3, (
                f"Three consecutive turns returned identical vehicle set {curr_vehicles} — system is looping"
            )
        else:
            consecutive_identical_recs = 0
        prev_vehicles = curr_vehicles

        replies.append(reply)

    assert session_id is not None, "session_id must persist across turns"
    discovery_replies = [r for r in replies if not any(v_id in r for v_id in ["#55", "#77", "#69", "#93"])]
    assert len(set(discovery_replies)) == len(discovery_replies) or len(discovery_replies) <= 1, (
        "Discovery replies should not repeat"
    )


def test_no_three_identical_recommendation_blocks(isolated_db: object) -> None:
    rag = _rag()
    session_id: str | None = None
    rec_blocks: list[str] = []

    for msg in [
        "family car, 4 people, budget 75000, family trips",
        "show me options",
        "something else please",
        "other options",
    ]:
        request = ChatRequest(message=msg, session_id=session_id)
        response: ChatResponse = handle_chat(request, rag=rag)
        session_id = response.session_id
        if response.vehicles:
            rec_blocks.append("|".join(str(v.id) for v in sorted(response.vehicles, key=lambda v: v.id)))

    if len(rec_blocks) >= 3:
        first_three = rec_blocks[:3]
        assert len(set(first_three)) > 1, (
            f"Three consecutive recommendation blocks are identical: {first_three}"
        )
