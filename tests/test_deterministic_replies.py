from unittest.mock import patch

from backend.orchestrator import ChatRequest, handle_chat
from backend.rag_service import PolicyRAGService


@patch("backend.orchestrator.synthesize_reply")
def test_inventory_search_skips_llm_synthesis(mock_synth, isolated_db):
    mock_synth.return_value = "hallucinated inventory"
    response = handle_chat(
        ChatRequest(message="Show me Tesla cars in inventory"),
        rag=PolicyRAGService(use_embeddings=False),
    )
    mock_synth.assert_not_called()
    assert "Tesla" in response.reply
    assert response.reply != "hallucinated inventory"


@patch("backend.orchestrator.synthesize_reply")
def test_hybrid_skips_llm_synthesis(mock_synth, isolated_db):
    mock_synth.return_value = "hallucinated hybrid"
    response = handle_chat(
        ChatRequest(message="BMW inventory and shipping policy"),
        rag=PolicyRAGService(use_embeddings=False),
    )
    mock_synth.assert_not_called()
    assert response.intent.value == "hybrid_rag"
