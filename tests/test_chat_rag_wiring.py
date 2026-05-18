from backend.main import app, get_rag_service
from backend.rag_service import PolicyRAGService


def test_chat_endpoint_uses_injected_rag_service(api_client):
    searches: list[str] = []

    class TrackingRAG(PolicyRAGService):
        def search(self, query: str, top_k: int = 3):
            searches.append(query)
            return super().search(query, top_k=top_k)

    tracking = TrackingRAG(use_embeddings=False)
    app.dependency_overrides[get_rag_service] = lambda: tracking
    try:
        response = api_client.post(
            "/api/chat",
            json={"message": "What is your refund policy for deposits?"},
        )
        assert response.status_code == 200
        assert searches
        assert response.json()["policy_context_used"] is True
    finally:
        app.dependency_overrides.clear()
