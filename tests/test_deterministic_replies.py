from pydantic import EmailStr

from backend.conversation_state import DialoguePhase
from backend.orchestrator import ChatRequest, handle_chat
from backend.rag_service import PolicyRAGService


def test_inventory_search_uses_db_text_only(isolated_db):
    response = handle_chat(
        ChatRequest(message="Show me Tesla cars in inventory"),
        rag=PolicyRAGService(use_embeddings=False),
    )
    assert "Tesla" in response.reply
    assert len(response.vehicles) > 0


def test_hybrid_uses_structured_context(isolated_db):
    response = handle_chat(
        ChatRequest(message="BMW inventory and shipping policy"),
        rag=PolicyRAGService(use_embeddings=False),
    )
    assert response.intent.value == "hybrid_rag"


def test_policy_reply_from_rag_chunks(isolated_db):
    response = handle_chat(
        ChatRequest(message="What is your refund policy for deposits?"),
        rag=PolicyRAGService(use_embeddings=False),
    )
    assert response.policy_context_used is True
    assert "refund" in response.reply.lower()


def test_general_chat_starts_sales_conversation(isolated_db):
    response = handle_chat(
        ChatRequest(message="Hello, what can you help me with?"),
        rag=PolicyRAGService(use_embeddings=False),
    )
    assert response.session_id
    assert response.dialogue_phase == DialoguePhase.DISCOVERY
    assert "?" in response.reply
    assert response.intent.value == "general_chat"
