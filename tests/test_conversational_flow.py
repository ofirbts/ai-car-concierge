from backend.conversation_state import ConversationState, save_conversation_state
from backend.conversation_state import ConversationState, DialoguePhase, save_conversation_state
from backend.orchestrator import ChatRequest, handle_chat
from backend.rag_service import PolicyRAGService


def _rag():
    return PolicyRAGService(use_embeddings=False)


def test_family_purchase_flow_multi_turn(isolated_db):
    session_id = None
    rag = _rag()

    r1 = handle_chat(
        ChatRequest(message="I'm looking for a family car", session_id=session_id),
        rag=rag,
    )
    session_id = r1.session_id
    assert session_id
    assert r1.dialogue_phase == DialoguePhase.DISCOVERY
    assert "?" in r1.reply

    r2 = handle_chat(
        ChatRequest(message="four people usually", session_id=session_id),
        rag=rag,
    )
    assert r2.session_id == session_id
    assert "budget" in r2.reply.lower() or "?" in r2.reply

    r3 = handle_chat(
        ChatRequest(message="around 75000", session_id=session_id),
        rag=rag,
    )
    assert r3.session_id == session_id

    r4 = handle_chat(
        ChatRequest(message="family trips, need space", session_id=session_id),
        rag=rag,
    )
    assert r4.session_id == session_id
    if r4.vehicles:
        assert r4.dialogue_phase in (DialoguePhase.RECOMMENDING, DialoguePhase.DISCOVERY)
        assert len(r4.vehicles) <= 4


def test_natural_language_family_query_returns_recommendations(isolated_db):
    from backend.inventory_retrieval import clear_inventory_retrieval_cache

    clear_inventory_retrieval_cache()
    response = handle_chat(
        ChatRequest(message="I need a good car for my family with kids"),
        rag=_rag(),
    )
    assert response.session_id
    assert response.dialogue_phase == DialoguePhase.DISCOVERY
    assert "?" in response.reply or response.vehicles


def test_compare_flow_with_session(isolated_db):
    session_id = handle_chat(
        ChatRequest(message="help me find an SUV under 80000"),
        rag=_rag(),
    ).session_id

    rec = handle_chat(
        ChatRequest(message="family of 4, budget 80000, city and highway", session_id=session_id),
        rag=_rag(),
    )
    assert rec.session_id == session_id

    if rec.last_recommended_ids if hasattr(rec, "last_recommended_ids") else rec.vehicles:
        ids = [v.id for v in rec.vehicles[:2]]
        if len(ids) >= 2:
            cmp_resp = handle_chat(
                ChatRequest(message=f"compare #{ids[0]} vs #{ids[1]}", session_id=session_id),
                rag=_rag(),
            )
            assert cmp_resp.dialogue_phase == DialoguePhase.COMPARING
            assert len(cmp_resp.reply) > 20


def test_reserve_within_sales_session(isolated_db):
    session_id = handle_chat(
        ChatRequest(message="looking for family SUV"),
        rag=_rag(),
    ).session_id

    handle_chat(
        ChatRequest(message="4 people, budget 90000", session_id=session_id),
        rag=_rag(),
    )
    handle_chat(
        ChatRequest(message="family use, prefer space", session_id=session_id),
        rag=_rag(),
    )

    reserve = handle_chat(
        ChatRequest(message="reserve vehicle #16", session_id=session_id),
        rag=_rag(),
    )
    assert reserve.intent.value == "reserve_intent"
    assert reserve.reserved_vehicle is not None
    assert reserve.reserved_vehicle.id == 16


def test_session_persistence_across_turns(isolated_db):
    from backend.conversation_state import load_conversation_state

    session_id = handle_chat(
        ChatRequest(message="I'm not sure what car I want"),
        rag=_rag(),
    ).session_id

    handle_chat(
        ChatRequest(message="couple with one kid", session_id=session_id),
        rag=_rag(),
    )

    stored = load_conversation_state(session_id)
    assert stored is not None
    assert stored.turn_count >= 2
    assert stored.passengers == 3


def test_explicit_tesla_search_stays_deterministic(isolated_db):
    response = handle_chat(
        ChatRequest(message="Show me Tesla cars in inventory"),
        rag=_rag(),
    )
    assert response.intent.value == "inventory_search"
    assert len(response.vehicles) > 0
    assert "Tesla" in response.reply


def test_grounded_recommendation_mentions_vehicle_facts(isolated_db):
    from backend.inventory_retrieval import clear_inventory_retrieval_cache

    clear_inventory_retrieval_cache()
    state = ConversationState(session_id="grounded-test")
    state.turn_count = 3
    state.budget = 70000
    state.passengers = 4
    state.body_type = "suv"
    state.use_case = "family trips"
    save_conversation_state(state)

    response = handle_chat(
        ChatRequest(message="show me what fits", session_id=state.session_id),
        rag=_rag(),
    )
    assert response.vehicles
    for vehicle in response.vehicles:
        assert str(vehicle.id) in response.reply or vehicle.make in response.reply


def test_policy_question_bypasses_sales_dialogue(isolated_db):
    response = handle_chat(
        ChatRequest(message="What is your refund policy for deposits?"),
        rag=_rag(),
    )
    assert response.intent.value == "policy_question"
    assert response.policy_context_used is True


def test_city_commute_semantic_search(isolated_db):
    from backend.inventory_retrieval import clear_inventory_retrieval_cache

    clear_inventory_retrieval_cache()
    response = handle_chat(
        ChatRequest(message="something economical for city driving"),
        rag=_rag(),
    )
    assert response.session_id
    assert response.dialogue_phase == DialoguePhase.DISCOVERY
