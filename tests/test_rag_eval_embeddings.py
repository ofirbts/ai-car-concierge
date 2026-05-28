import json
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.rag_service import PolicyRAGService

GOLDEN_PATH = Path(__file__).resolve().parent / "fixtures" / "rag_golden.json"


@pytest.fixture
def embedding_rag():
    return PolicyRAGService(use_embeddings=True)


def test_rag_golden_recall_with_mocked_embeddings(embedding_rag):
    cases = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    with patch("backend.rag_service.embed_texts") as mock_embed, patch(
        "backend.rag_service.embed_query", return_value=[1.0, 0.0, 0.0]
    ):
        mock_embed.return_value = [[1.0, 0.0, 0.0] for _ in range(50)]
        for case in cases:
            result = embedding_rag.search(case["query"], top_k=3)
            assert result.chunks, f"No chunks for: {case['query']}"
            assert result.retrieval_mode == "gemini_embeddings"
            if "must_source" in case:
                assert any(
                    case["must_source"] in chunk.source for chunk in result.chunks
                ), case["query"]


def test_refund_deposit_prefers_faq_source_with_equal_embeddings(embedding_rag):
    with patch("backend.rag_service.embed_texts") as mock_embed, patch(
        "backend.rag_service.embed_query", return_value=[1.0, 0.0, 0.0]
    ):
        mock_embed.return_value = [[1.0, 0.0, 0.0] for _ in range(len(embedding_rag._chunks))]
        result = embedding_rag.search("refund policy deposit", top_k=1)
        assert result.chunks
        assert result.chunks[0].source == "faqs.md"
