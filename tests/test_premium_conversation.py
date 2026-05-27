from backend.conversation_state import ConversationState
from backend.conversational_nlg import generate_comparison, generate_recommendations
from backend.database import VehicleSearchFilters, search_vehicles
from backend.intent import classify_intent_rule_based
from backend.rag_service import PolicyRAGService
from backend.sales_dialogue import handle_sales_turn


def _family_state() -> ConversationState:
    state = ConversationState(session_id="premium")
    state.turn_count = 3
    state.passengers = 4
    state.family_size = 4
    state.budget = 75000
    state.body_type = "suv"
    state.use_case = "family trips"
    state.space_priority = "space"
    return state


def test_recommendations_are_premium_and_non_template(isolated_db):
    state = _family_state()
    vehicles = search_vehicles(VehicleSearchFilters(year_min=2022, in_stock_only=True, limit=3))
    text = generate_recommendations(state, vehicles[:3]).lower()
    assert "fits your budget" not in text
    assert "premium option" not in text
    assert "shortlist" in text or "strongest" in text


def test_comparison_uses_tradeoff_reasoning(isolated_db):
    state = _family_state()
    vehicles = search_vehicles(VehicleSearchFilters(year_min=2022, in_stock_only=True, limit=3))
    text = generate_comparison(state, vehicles[:3]).lower()
    assert "value play" in text or "trade" in text or "frame it this way" in text
    assert "city" in text or "comfort" in text


def test_budget_objection_refines_recommendation(isolated_db):
    state = _family_state()
    state.last_recommended_ids = [55, 77, 69]
    extracted = classify_intent_rule_based("these are too expensive, show me cheaper options")
    turn = handle_sales_turn(
        "these are too expensive, show me cheaper options",
        extracted,
        state,
        None,
        PolicyRAGService(use_embeddings=False),
    )
    assert turn.rag_mode == "sales_dialogue+budget_objection"
    assert "value" in turn.reply.lower() or "cheaper" in turn.reply.lower()
    assert len(turn.vehicles) <= 3
