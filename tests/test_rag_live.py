import json
import os
from pathlib import Path

import pytest

from backend.rag_service import PolicyRAGService

GOLDEN_PATH = Path(__file__).resolve().parent / "fixtures" / "rag_golden.json"


@pytest.mark.live
def test_policy_rag_live_embeddings_recall():
    if not os.environ.get("GOOGLE_API_KEY", "").strip():
        pytest.skip("GOOGLE_API_KEY not set")
    rag = PolicyRAGService(use_embeddings=True)
    cases = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))[:3]
    for case in cases:
        result = rag.search(case["query"], top_k=3)
        assert result.chunks, case["query"]
        assert result.retrieval_mode == "gemini_embeddings"
        if "must_source" in case:
            assert any(
                case["must_source"] in chunk.source for chunk in result.chunks
            ), case["query"]
