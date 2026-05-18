from unittest.mock import patch

from backend.rag_service import PolicyRAGService


def test_gemini_embedding_path_uses_embed_texts(tmp_path):
    policies = tmp_path / "policies"
    policies.mkdir()
    (policies / "policy.md").write_text(
        "# Policy\n\n## Sales\n\nOnly 2022 and newer.\n\n## Returns\n\nSee support.\n",
        encoding="utf-8",
    )

    with patch("backend.rag_service.embed_texts") as mock_embed, patch(
        "backend.rag_service.embed_query", return_value=[1.0, 0.0]
    ):
        mock_embed.return_value = [[1.0, 0.0], [0.0, 1.0]]
        service = PolicyRAGService(policies_dir=policies, use_embeddings=True)
        result = service.search("2022 sales", top_k=2)

    assert len(result.chunks) == 2
    assert result.retrieval_mode == "gemini_embeddings"
    mock_embed.assert_called_once()


def test_search_returns_requested_top_k_even_with_low_scores(tmp_path):
    policies = tmp_path / "policies"
    policies.mkdir()
    (policies / "a.md").write_text("# A\n\n## One\n\nalpha beta.\n", encoding="utf-8")
    (policies / "b.md").write_text("# B\n\n## Two\n\ngamma delta.\n", encoding="utf-8")

    service = PolicyRAGService(policies_dir=policies, use_embeddings=False)
    result = service.search("zzzzunknown", top_k=2)
    assert len(result.chunks) == 2
