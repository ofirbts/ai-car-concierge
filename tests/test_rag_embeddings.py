from unittest.mock import MagicMock

from backend.rag_service import PolicyRAGService


def test_openai_embedding_path_uses_mock_client(tmp_path):
    policies = tmp_path / "policies"
    policies.mkdir()
    (policies / "policy.md").write_text(
        "# Policy\n\n## Sales\n\nOnly 2022 and newer.\n\n## Returns\n\nSee support.\n",
        encoding="utf-8",
    )

    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = [
        MagicMock(data=[MagicMock(embedding=[1.0, 0.0]), MagicMock(embedding=[0.0, 1.0])]),
        MagicMock(data=[MagicMock(embedding=[1.0, 0.0])]),
    ]

    service = PolicyRAGService(
        policies_dir=policies,
        use_openai=True,
        client=mock_client,
    )
    result = service.search("2022 sales", top_k=2)
    assert len(result.chunks) == 2
    assert mock_client.embeddings.create.call_count == 2


def test_search_returns_requested_top_k_even_with_low_scores(tmp_path):
    policies = tmp_path / "policies"
    policies.mkdir()
    (policies / "a.md").write_text("# A\n\n## One\n\nalpha beta.\n", encoding="utf-8")
    (policies / "b.md").write_text("# B\n\n## Two\n\ngamma delta.\n", encoding="utf-8")

    service = PolicyRAGService(policies_dir=policies, use_openai=False)
    result = service.search("zzzzunknown", top_k=2)
    assert len(result.chunks) == 2
