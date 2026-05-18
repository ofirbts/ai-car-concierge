import pytest

from backend.database import PolicyViolationError, assert_sellable, get_vehicle_by_id, reserve_vehicle
from backend.intent import IntentKind, classify_intent_rule_based
from backend.orchestrator import ChatRequest, handle_chat
from backend.rag_service import PolicyRAGService


def test_classify_reserve_intent():
    intent = classify_intent_rule_based("please reserve vehicle #16")
    assert intent.intent == IntentKind.RESERVE_INTENT
    assert intent.vehicle_id == 16


def test_classify_policy_intent():
    intent = classify_intent_rule_based("how do refunds work?")
    assert intent.intent == IntentKind.POLICY_QUESTION


def test_assert_sellable_blocks_pre_2022(isolated_db):
    vehicle = get_vehicle_by_id(5)
    assert vehicle is not None
    assert vehicle.pending_delisting
    with pytest.raises(PolicyViolationError):
        assert_sellable(vehicle)


def test_reserve_vehicle_raises_policy(isolated_db):
    with pytest.raises(PolicyViolationError):
        reserve_vehicle(5)


def test_handle_chat_notes_delisted_inventory(isolated_db):
    response = handle_chat(
        ChatRequest(message="show cars from 2019"),
        rag=PolicyRAGService(use_openai=False),
    )
    assert response.intent == IntentKind.LEGACY_YEAR_CONFLICT
    assert response.blocked is True
    assert any(v.pending_delisting for v in response.vehicles)
    assert "De-listing" in response.reply or "2022" in response.reply
