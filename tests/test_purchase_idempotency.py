from unittest.mock import patch

from backend.automations import EmailResult
from backend.database import try_claim_purchase_notification
from backend.orchestrator import ChatRequest, handle_chat
from backend.rag_service import PolicyRAGService


def test_try_claim_purchase_notification_once(isolated_db):
    assert try_claim_purchase_notification("pk-1", "a@b.com", 16) is True
    assert try_claim_purchase_notification("pk-1", "a@b.com", 16) is False


@patch("backend.orchestrator.send_purchase_inquiry_email")
def test_purchase_inquiry_idempotent(mock_send, isolated_db):
    mock_send.return_value = EmailResult(sent=True)
    payload = ChatRequest(
        message="I want to buy a family SUV",
        user_email="buyer@example.com",
        idempotency_key="purchase-inq-1",
    )
    first = handle_chat(payload, rag=PolicyRAGService(use_embeddings=False))
    second = handle_chat(payload, rag=PolicyRAGService(use_embeddings=False))
    assert first.email_sent is True
    assert second.email_sent is False
    assert "already received" in second.reply.lower()
    mock_send.assert_called_once()
