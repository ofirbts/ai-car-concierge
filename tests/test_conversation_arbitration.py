from __future__ import annotations

from backend.conversation_state import ConversationState, new_session_id
from backend.conversation_understanding import (
    ConvIntent,
    understand_conversation,
    _regex_understand,
)
from backend.dialogue_policy import PolicyAction, decide_policy
from backend.intent import classify_intent_rule_based
from backend.rag_service import PolicyRAGService
from backend.sales_dialogue import _update_state_from_understanding, handle_sales_turn


def _state(**kwargs) -> ConversationState:
    s = ConversationState(session_id=new_session_id())
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


def test_product_question_not_passenger_slot() -> None:
    u = _regex_understand("מה אתה עושה?")
    assert u.conv_intent == ConvIntent.PRODUCT_EXPLANATION
    d = decide_policy(_state(), u, "מה אתה עושה?")
    assert d.action == PolicyAction.EXPLAIN_PRODUCT


def test_criteria_what_else_you_check() -> None:
    u = _regex_understand("איזה עוד דברים אתה בודק?")
    assert u.conv_intent == ConvIntent.CRITERIA_INQUIRY
    d = decide_policy(_state(), u, "איזה עוד דברים אתה בודק?")
    assert d.action == PolicyAction.EXPLAIN_CRITERIA


def test_passenger_one_when_asked_passengers_not_budget_clarify() -> None:
    state = _state(last_asked_field="passengers")
    u = understand_conversation("1", state)
    assert u.slots.passengers == 1
    assert u.slots.budget is None
    _update_state_from_understanding(
        state, u, "1", classify_intent_rule_based("1"), None
    )
    assert state.passengers == 1
    d = decide_policy(state, u, "1")
    assert d.action != PolicyAction.CLARIFY_BUDGET
    assert d.action == PolicyAction.ASK_USE_CASE


def test_budget_number_only_when_asked_budget() -> None:
    state = _state(last_asked_field="budget")
    u = understand_conversation("45000", state)
    assert u.slots.budget == 45000.0


def test_user_correction_after_misread() -> None:
    state = _state(last_asked_field="passengers", budget=1.0)
    u = _regex_understand("אני הקלדתי את הספרה 1 איך הגעת לשאלה הזו?", state)
    assert u.conv_intent == ConvIntent.USER_CORRECTION
    d = decide_policy(state, u, "אני הקלדתי את הספרה 1 איך הגעת לשאלה הזו?")
    assert d.action == PolicyAction.REPAIR_TURN
    assert d.question_hint == "misunderstood_slot"


def test_no_limit_unblocks_budget_discovery() -> None:
    state = _state(last_asked_field="budget")
    u = understand_conversation("no limit", state)
    assert u.slots.budget_unconstrained is True
    s = _state()
    s.last_asked_field = "budget"
    from backend.sales_dialogue import _update_state_from_understanding

    _update_state_from_understanding(s, u, "no limit", classify_intent_rule_based("no limit"), None)
    assert s.budget_unconstrained is True
    assert s.has_discovery_basics() or s.budget_unconstrained


def test_negative_feedback_triggers_repair_not_recommendation(isolated_db) -> None:
    state = _state(
        turn_count=5,
        passengers=4,
        budget=75000,
        use_case="family trips",
        body_type="suv",
        last_recommended_ids=[55, 77, 69],
    )
    turn = handle_sales_turn(
        "לא טוב",
        classify_intent_rule_based("לא טוב"),
        state,
        None,
        PolicyRAGService(use_embeddings=False),
    )
    assert turn.show_vehicle_cards is False
    assert turn.vehicles == []
    assert "מחיר" in turn.reply or "price" in turn.reply.lower()


def test_criteria_turn_no_vehicle_cards(isolated_db) -> None:
    state = _state(
        turn_count=6,
        passengers=1,
        use_case="family trips",
        budget=78888,
        last_recommended_ids=[55, 77, 69],
    )
    turn = handle_sales_turn(
        "איזה עוד דברים אתה בודק?",
        classify_intent_rule_based("איזה עוד דברים אתה בודק?"),
        state,
        None,
        PolicyRAGService(use_embeddings=False),
    )
    assert turn.show_vehicle_cards is False
    assert turn.vehicles == []
    assert "יעילות" in turn.reply or "efficiency" in turn.reply.lower()


def test_fuel_priority_recommendation_penalizes_gas_navigator(isolated_db) -> None:
    state = _state(
        turn_count=6,
        passengers=4,
        budget=90000,
        use_case="family trips",
        body_type="suv",
        last_recommended_ids=[55, 77, 69],
        space_priority="fuel",
    )
    turn = handle_sales_turn(
        "צריכת דלק",
        classify_intent_rule_based("צריכת דלק"),
        state,
        None,
        PolicyRAGService(use_embeddings=False),
    )
    assert turn.vehicles
    top_fuel = (turn.vehicles[0].fuel_type or "").lower()
    assert "electric" in top_fuel or "hybrid" in top_fuel or "plug" in top_fuel
