from backend.intent import IntentKind, classify_intent_rule_based
from backend.orchestrator import ChatRequest, handle_chat
from backend.rag_service import PolicyRAGService


def test_classify_hybrid_intent_without_legacy_year():
    intent = classify_intent_rule_based(
        "BMW X5 price and what is your refund policy?"
    )
    assert intent.intent == IntentKind.HYBRID_RAG


def test_model_3_alias_sets_tesla(isolated_db):
    intent = classify_intent_rule_based("Model 3 price and refund policy")
    assert intent.intent == IntentKind.HYBRID_RAG
    assert intent.make == "Tesla"
    assert intent.model == "Model 3"


def test_hybrid_model_3_inventory_filtered(isolated_db):
    response = handle_chat(
        ChatRequest(message="Model 3 price and refund policy"),
        rag=PolicyRAGService(use_embeddings=False),
    )
    assert response.intent == IntentKind.HYBRID_RAG
    assert all("Tesla" in v.make for v in response.vehicles)


def test_classify_hybrid_with_2020_becomes_legacy_conflict():
    intent = classify_intent_rule_based(
        "2020 Tesla price and what is your refund policy?"
    )
    assert intent.intent == IntentKind.LEGACY_YEAR_CONFLICT


def test_classify_legacy_year_intent():
    intent = classify_intent_rule_based("Do you have a 2020 Tesla in stock?")
    assert intent.intent == IntentKind.LEGACY_YEAR_CONFLICT


def test_handle_legacy_year_conflict(isolated_db):
    response = handle_chat(
        ChatRequest(message="Any 2021 Audi A4 available?"),
        rag=PolicyRAGService(use_embeddings=False),
    )
    assert response.intent == IntentKind.LEGACY_YEAR_CONFLICT
    assert response.blocked is True
    assert "2022" in response.reply or "De-listing" in response.reply
    assert "2021" in response.reply or response.vehicles


def test_hybrid_legacy_year_uses_conflict_handler(isolated_db):
    response = handle_chat(
        ChatRequest(message="2020 Tesla price and refund policy"),
        rag=PolicyRAGService(use_embeddings=False),
    )
    assert response.intent == IntentKind.LEGACY_YEAR_CONFLICT
    assert response.blocked is True


def test_hybrid_returns_inventory_and_policy(isolated_db):
    response = handle_chat(
        ChatRequest(message="BMW X5 price and shipping delivery cost"),
        rag=PolicyRAGService(use_embeddings=False),
    )
    assert response.intent == IntentKind.HYBRID_RAG
    assert response.policy_context_used is True


def test_api_hybrid_chat(api_client):
    response = api_client.post(
        "/api/chat",
        json={"message": "Tesla inventory and test drive scheduling policy"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "hybrid_rag"
    assert data["policy_context_used"] is True


def test_api_legacy_2020_conflict(api_client):
    response = api_client.post(
        "/api/chat",
        json={"message": "I want a 2020 Jaguar F-PACE"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "legacy_year_conflict"
    assert data["blocked"] is True
