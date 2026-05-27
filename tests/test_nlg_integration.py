from unittest.mock import patch


def test_chat_recommendation_uses_fallback_when_llm_invents_price(isolated_db, api_client):
    with patch(
        "backend.conversational_nlg.generate_text",
        return_value="I would shortlist this beauty today for only $99,999.",
    ):
        welcome = api_client.post("/api/chat", json={"message": "I'm looking for a family car"})
        assert welcome.status_code == 200
        session_id = welcome.json()["session_id"]
        assert session_id

        turn = api_client.post(
            "/api/chat",
            json={
                "message": "four people, budget 75000, family trips, need an SUV",
                "session_id": session_id,
            },
        )
    assert turn.status_code == 200
    data = turn.json()
    assert "$99,999" not in data["reply"]
    assert data.get("vehicles")
    assert data["validation_verdict"] == "PASS"
