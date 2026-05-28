from backend.conversation_state import ConversationState
from backend.database import search_vehicles, VehicleSearchFilters, SALES_MIN_YEAR
from backend.intent import classify_intent_rule_based
from backend.inventory_retrieval import clear_inventory_retrieval_cache, family_fit_score, hybrid_search_inventory
from backend.orchestrator import ChatRequest, handle_chat
from backend.rag_service import PolicyRAGService
from backend.sales_dialogue import handle_sales_turn, update_state_from_message


def _rag():
    return PolicyRAGService(use_embeddings=False)


def test_family_scoring_prefers_navigator_over_cayenne(isolated_db):
    clear_inventory_retrieval_cache()
    state = ConversationState(session_id="fam", passengers=4, use_case="family trips", budget=75000)
    candidates = search_vehicles(
        VehicleSearchFilters(year_min=SALES_MIN_YEAR, in_stock_only=True, limit=100)
    )
    nav = next((v for v in candidates if "Navigator" in v.model), None)
    cayenne = next((v for v in candidates if v.model == "Cayenne"), None)
    assert nav is not None
    assert cayenne is not None
    assert family_fit_score(nav, state) > family_fit_score(cayenne, state)


def test_hybrid_search_family_avoids_cayenne_first(isolated_db):
    clear_inventory_retrieval_cache()
    state = ConversationState(
        session_id="fam2",
        passengers=4,
        use_case="family trips",
        budget=75000,
        body_type="suv",
    )
    result = hybrid_search_inventory("family SUV under 75000", state=state, limit=4)
    assert result.vehicles
    top_models = [v.model for v in result.vehicles[:2]]
    assert not all("Cayenne" in m for m in top_models)


def test_preference_refinement_does_not_repeat_full_recommendation(isolated_db):
    state = ConversationState(session_id="refine")
    state.turn_count = 2
    state.passengers = 4
    state.budget = 75000
    state.use_case = "family trips"
    state.body_type = "suv"
    state.last_recommended_ids = [55, 92, 77, 76]
    state.space_priority = "space"

    extracted = classify_intent_rule_based("need space for family trips")
    turn = handle_sales_turn(
        "need space for family trips",
        extracted,
        state,
        None,
        _rag(),
    )
    assert "shortlist" in turn.reply.lower() or "space" in turn.reply.lower()
    assert turn.rag_mode == "sales_dialogue+preference_refine"
    assert "here are a few options" not in turn.reply.lower()


def test_reserve_outside_shortlist_gets_note(isolated_db):
    session_id = handle_chat(
        ChatRequest(message="I'm looking for a family car"),
        rag=_rag(),
    ).session_id
    handle_chat(
        ChatRequest(message="four people, budget 90000", session_id=session_id),
        rag=_rag(),
    )
    handle_chat(
        ChatRequest(message="family trips, need space", session_id=session_id),
        rag=_rag(),
    )
    response = handle_chat(
        ChatRequest(message="reserve vehicle #16", session_id=session_id),
        rag=_rag(),
    )
    assert response.reserved_vehicle is not None
    if response.reserved_vehicle.id not in (response.vehicles or []):
        assert "wasn't in your latest recommendations" in response.reply or "Done" in response.reply


def test_smalltalk_does_not_repeat_recommendations(isolated_db):
    state = ConversationState(session_id="smalltalk")
    state.turn_count = 4
    state.passengers = 4
    state.budget = 75000
    state.use_case = "family trips"
    state.body_type = "suv"
    state.last_recommended_ids = [55, 77, 69]
    extracted = classify_intent_rule_based("how old are u")
    turn = handle_sales_turn(
        "how old are u",
        extracted,
        state,
        None,
        _rag(),
    )
    assert turn.intent.value == "general_chat"
    assert turn.show_vehicle_cards is False
    assert turn.vehicles == []
    assert "advisor" in turn.reply.lower()
