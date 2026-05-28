from backend.conversation_state import ConversationState
from backend.dialogue_analysis import analyze_dialogue_turn
from backend.dialogue_policy import choose_dialogue_policy
from backend.intent import classify_intent_rule_based
from backend.rag_service import PolicyRAGService
from backend.sales_dialogue import handle_sales_turn


def test_dialogue_analysis_detects_smalltalk():
    state = ConversationState(session_id="d1")
    analysis = analyze_dialogue_turn("how old are u", state)
    assert analysis.smalltalk is True
    assert analysis.intent_label == "smalltalk"


def test_policy_engine_triggers_repair_on_frustration():
    state = ConversationState(session_id="d2")
    analysis = analyze_dialogue_turn("this is bad and not good", state)
    decision = choose_dialogue_policy(state, analysis)
    assert decision.action == "repair_turn"
    assert decision.question


def test_topic_shift_recalibrates_without_recommendation_cards(isolated_db):
    state = ConversationState(session_id="d3")
    state.turn_count = 4
    state.passengers = 4
    state.budget = 75000
    state.use_case = "family trips"
    state.body_type = "suv"
    state.last_recommended_ids = [55, 77, 69]
    extracted = classify_intent_rule_based("something else, not family")
    turn = handle_sales_turn(
        "something else, not family",
        extracted,
        state,
        None,
        PolicyRAGService(use_embeddings=False),
    )
    assert turn.intent.value == "general_chat"
    assert turn.show_vehicle_cards is False
    assert "optimize" in turn.reply.lower() or "what should" in turn.reply.lower()
    assert state.last_recommended_ids == []


def test_language_switch_to_hebrew(isolated_db):
    state = ConversationState(session_id="d4")
    extracted = classify_intent_rule_based("answer in hebrew")
    turn = handle_sales_turn(
        "answer in hebrew",
        extracted,
        state,
        None,
        PolicyRAGService(use_embeddings=False),
    )
    assert state.language_preference == "he"
    assert turn.show_vehicle_cards is False
    assert "בעברית" in turn.reply


def test_unclear_followup_triggers_clarify_constraints(isolated_db):
    state = ConversationState(session_id="d5")
    state.turn_count = 5
    state.passengers = 4
    state.budget = 75000
    state.use_case = "family trips"
    state.body_type = "suv"
    state.last_recommended_ids = [55, 77, 69]
    extracted = classify_intent_rule_based("no")
    turn = handle_sales_turn(
        "no",
        extracted,
        state,
        None,
        PolicyRAGService(use_embeddings=False),
    )
    assert turn.intent.value == "general_chat"
    assert turn.show_vehicle_cards is False
    assert "what should i change first" in turn.reply.lower()
