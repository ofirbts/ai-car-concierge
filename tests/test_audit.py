from backend.database import count_audit_rows
from backend.orchestrator import ChatRequest, handle_chat
from backend.rag_service import PolicyRAGService


def test_reserve_writes_audit_row(isolated_db):
    before = count_audit_rows()
    handle_chat(
        ChatRequest(message="reserve vehicle #16", idempotency_key="audit-res-1"),
        rag=PolicyRAGService(use_embeddings=False),
    )
    assert count_audit_rows() == before + 1
