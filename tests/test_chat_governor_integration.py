import os
from pathlib import Path


def test_chat_governor_writes_classify_route_validate_steps(api_client):
    response = api_client.post(
        "/api/chat",
        json={"message": "show me tesla under 70000", "user_email": "test@example.com"},
    )
    assert response.status_code == 200

    journal_path = Path(os.environ["CHAT_GOVERNOR_JOURNAL"])
    assert journal_path.is_file()
    content = journal_path.read_text(encoding="utf-8")
    assert "classify_intent" in content
    assert "route_response" in content
    assert "validate_response" in content
