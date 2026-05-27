from backend.conversation_state import ConversationState
from backend.conversational_nlg import _dedupe_sentences, _fallback_recommendations, polish_response
from backend.database import VehicleSearchFilters, search_vehicles
from backend.intent import classify_intent_rule_based
from backend.rag_service import PolicyRAGService
from backend.sales_dialogue import handle_sales_turn


def _family_state() -> ConversationState:
    state = ConversationState(session_id="copy")
    state.turn_count = 3
    state.passengers = 4
    state.budget = 75000
    state.body_type = "suv"
    state.use_case = "family trips"
    state.space_priority = "space"
    return state


def test_fallback_recommendations_use_distinct_reasons(isolated_db):
    state = _family_state()
    vehicles = search_vehicles(VehicleSearchFilters(year_min=2022, in_stock_only=True, limit=3))
    text = _fallback_recommendations(state, vehicles)
    lowered = text.lower()
    assert lowered.count("you will feel the extra room immediately on family drives") <= 1
    assert "navigator" in lowered or "#" in lowered


def test_dedupe_sentences_removes_repeated_lines():
    raw = "Hello there. Hello there. Next point."
    assert _dedupe_sentences(raw) == "Hello there. Next point."


def test_preference_refinement_skips_duplicate_cards(isolated_db):
    state = ConversationState(session_id="refine-cards")
    state.turn_count = 4
    state.passengers = 4
    state.budget = 75000
    state.use_case = "family trips"
    state.body_type = "suv"
    state.space_priority = "space"
    state.last_recommended_ids = [55, 77, 69]
    state.last_refinement_key = "space"
    extracted = classify_intent_rule_based("need space for family trips")
    turn = handle_sales_turn(
        "need space for family trips",
        extracted,
        state,
        None,
        PolicyRAGService(use_embeddings=False),
    )
    assert turn.show_vehicle_cards is False
    assert turn.vehicles == []
    assert "aligned" in turn.reply.lower()


def test_compare_turn_hides_vehicle_cards(isolated_db):
    state = _family_state()
    state.last_recommended_ids = [55, 77, 69]
    extracted = classify_intent_rule_based("what's the best value here?")
    turn = handle_sales_turn(
        "what's the best value here?",
        extracted,
        state,
        None,
        PolicyRAGService(use_embeddings=False),
    )
    assert turn.show_vehicle_cards is False
    assert "value" in turn.reply.lower() or "frame" in turn.reply.lower()
