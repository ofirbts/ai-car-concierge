import json
from pathlib import Path

import pytest

from backend.rag_service import PolicyRAGService

GOLDEN_PATH = Path(__file__).resolve().parent / "fixtures" / "rag_golden.json"


@pytest.fixture
def keyword_rag():
    return PolicyRAGService(use_embeddings=False)


def test_rag_golden_recall(keyword_rag):
    cases = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    for case in cases:
        result = keyword_rag.search(case["query"], top_k=3)
        assert result.chunks, f"No chunks for: {case['query']}"
        if "must_source" in case:
            assert any(
                case["must_source"] in chunk.source for chunk in result.chunks
            ), case["query"]
        if "must_heading_substr" in case:
            assert any(
                case["must_heading_substr"].lower() in (chunk.heading or "").lower()
                for chunk in result.chunks
            ), case["query"]
