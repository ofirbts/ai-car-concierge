from backend.conversation_state import ConversationState
from backend.conversational_nlg import (
    _dedupe_sentences,
    _fallback_recommendations,
    _join_prose,
    _memory_callback,
    polish_response,
)
from backend.sales_dialogue import _contextual_discovery_question, _parse_use_case
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


def test_join_prose_adds_space_after_period():
    assert _join_prose("Hello.", "Next step.") == "Hello. Next step."


def test_polish_response_fixes_period_glue():
    assert ".Next" not in polish_response("drives.Next, the Lexus")


def test_contextual_budget_question_after_profile():
    state = ConversationState(session_id="ctx")
    state.passengers = 4
    state.use_case = "city and weekend drives"
    q = _contextual_discovery_question(state, "What's your target budget (roughly)?")
    assert "Got it" in q
    assert "60000" not in q
    assert "budget" in q.lower()


def test_use_case_prefers_city_weekend_over_family_keyword():
    assert _parse_use_case("family of 4, mostly city and weekend drives") == "city and weekend drives"


def test_memory_callback_uses_city_not_family_trips():
    state = ConversationState(session_id="mem")
    state.use_case = "city and weekend drives"
    assert "city" in _memory_callback(state).lower()
    assert "family trips" not in _memory_callback(state).lower()


def test_budget_objection_hides_cards_when_already_at_floor(isolated_db):
    state = ConversationState(session_id="budget-hide")
    state.turn_count = 5
    state.passengers = 4
    state.budget = 60000
    state.use_case = "city and weekend drives"
    state.body_type = "suv"
    cheapest = sorted(
        search_vehicles(VehicleSearchFilters(year_min=2022, in_stock_only=True, limit=80)),
        key=lambda v: v.price,
    )[:3]
    state.last_recommended_ids = [v.id for v in cheapest]
    extracted = classify_intent_rule_based("these feel expensive, can we go cheaper?")
    turn = handle_sales_turn(
        "these feel expensive, can we go cheaper?",
        extracted,
        state,
        None,
        PolicyRAGService(use_embeddings=False),
    )
    assert turn.rag_mode == "sales_dialogue+budget_objection"
    assert turn.show_vehicle_cards is False
    assert "already" in turn.reply.lower() or "shortlist" in turn.reply.lower()


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


def test_city_recommendation_avoids_robotic_balanced_phrase(isolated_db):
    state = ConversationState(session_id="city-natural")
    state.turn_count = 3
    state.passengers = 2
    state.budget = 75000
    state.use_case = "city driving"
    state.body_type = "sedan"
    vehicles = search_vehicles(VehicleSearchFilters(year_min=2022, in_stock_only=True, limit=3))
    text = _fallback_recommendations(state, vehicles).lower()
    assert "balanced match for how you plan to use the car" not in text
    assert "my first pick" in text
