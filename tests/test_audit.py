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


def test_reserve_via_api_writes_audit_row(api_client):
    before = count_audit_rows()
    response = api_client.post(
        "/api/chat",
        json={"message": "reserve vehicle #17", "idempotency_key": "audit-api-17"},
    )
    assert response.status_code == 200
    assert count_audit_rows() == before + 1
