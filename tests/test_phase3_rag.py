from pathlib import Path

import pytest

from backend.rag_service import PolicyRAGService, load_policy_chunks

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ROOT / "data" / "policies"


@pytest.fixture
def rag():
    return PolicyRAGService(policies_dir=POLICIES, use_openai=False)


def test_loads_all_five_policy_files():
    chunks = load_policy_chunks(POLICIES)
    sources = {c["source"] for c in chunks}
    assert sources == {
        "faqs.md",
        "maintenance.md",
        "policy.md",
        "shipping.md",
        "support.md",
    }
    assert len(chunks) >= 10


def test_refund_query_returns_support(rag):
    result = rag.search("refund deposit cancellation", top_k=2)
    assert result.chunks
    assert any("support.md" in c.source for c in result.chunks)
    assert any("refund" in c.content.lower() for c in result.chunks)


def test_test_drive_query_returns_faqs(rag):
    result = rag.search("schedule test drive appointment", top_k=2)
    assert result.chunks
    assert any("faqs.md" in c.source for c in result.chunks)


def test_ev_maintenance_query_returns_maintenance(rag):
    result = rag.search("electric EV battery annual service", top_k=2)
    assert result.chunks
    assert any("maintenance.md" in c.source for c in result.chunks)
    assert any(
        "electric" in c.heading.lower() or "ev" in c.content.lower()
        for c in result.chunks
    )


def test_shipping_query_returns_shipping(rag):
    result = rag.search("national delivery enclosed transport cost", top_k=2)
    assert result.chunks
    assert any("shipping.md" in c.source for c in result.chunks)


def test_sales_policy_query_returns_policy(rag):
    result = rag.search("2022 sales policy pending de-listing", top_k=2)
    assert result.chunks
    assert result.chunks[0].source == "policy.md"
    assert "2022" in result.chunks[0].content


def test_format_context_includes_source_and_body(rag):
    result = rag.search("refund", top_k=1)
    text = PolicyRAGService.format_context(result)
    assert "[support.md" in text
    assert "refund" in text.lower()


def test_empty_query_returns_no_chunks(rag):
    result = rag.search("  ", top_k=3)
    assert result.chunks == []
