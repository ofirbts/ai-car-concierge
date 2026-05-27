from backend.conversation_state import (
    ConversationState,
    DialoguePhase,
    get_or_create_state,
    load_conversation_state,
    save_conversation_state,
)
from backend.inventory_retrieval import (
    clear_inventory_retrieval_cache,
    detect_semantic_profiles,
    hybrid_search_inventory,
    infer_body_type,
)
from backend.sales_dialogue import (
    has_sales_signal,
    should_use_sales_dialogue,
    update_state_from_message,
)


def test_conversation_state_persistence(isolated_db):
    state = ConversationState(session_id="sess-1", budget=60000, passengers=4)
    state.phase = DialoguePhase.DISCOVERY
    save_conversation_state(state)
    loaded = load_conversation_state("sess-1")
    assert loaded is not None
    assert loaded.budget == 60000
    assert loaded.passengers == 4
    assert loaded.phase == DialoguePhase.DISCOVERY


def test_get_or_create_state_new_session(isolated_db):
    state = get_or_create_state(None)
    assert state.session_id
    assert state.turn_count == 0


def test_slot_extraction_family(isolated_db):
    state = ConversationState(session_id="sess-2")
    from backend.intent import classify_intent_rule_based

    extracted = classify_intent_rule_based("family car for two kids under 70000")
    update_state_from_message(state, "family car for two kids under 70000", extracted, None)
    assert state.passengers == 4
    assert state.budget == 70000
    assert state.body_type == "suv"


def test_semantic_profiles_detect_family(isolated_db):
    profiles = detect_semantic_profiles("good car for my family")
    assert "family" in profiles


def test_hybrid_search_finds_sellable_vehicles(isolated_db):
    clear_inventory_retrieval_cache()
    state = ConversationState(session_id="s", budget=80000, body_type="suv")
    result = hybrid_search_inventory("family SUV not too expensive", state=state, limit=4)
    assert result.vehicles
    assert all(not v.pending_delisting for v in result.vehicles)
    assert all(v.year >= 2022 for v in result.vehicles)


def test_infer_body_type_suv(isolated_db):
    from backend.database import get_vehicle_by_id

    vehicle = get_vehicle_by_id(17)
    assert vehicle is not None
    assert infer_body_type(vehicle) == "suv"


def test_should_use_sales_dialogue_general_chat(isolated_db):
    from backend.intent import ExtractedIntent, IntentKind

    extracted = ExtractedIntent(intent=IntentKind.GENERAL_CHAT)
    assert should_use_sales_dialogue(extracted, "hello", None, None) is True


def test_should_not_use_sales_for_explicit_tesla_search(isolated_db):
    from backend.intent import ExtractedIntent, IntentKind

    extracted = ExtractedIntent(intent=IntentKind.INVENTORY_SEARCH, make="Tesla")
    assert should_use_sales_dialogue(extracted, "Show me Tesla cars in inventory", None, None) is False


def test_budget_around_parsing(isolated_db):
    state = ConversationState(session_id="sess-budget")
    from backend.intent import classify_intent_rule_based

    extracted = classify_intent_rule_based("budget around 75000")
    update_state_from_message(state, "budget around 75000", extracted, None)
    assert state.budget == 75000
