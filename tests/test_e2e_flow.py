def test_reserve_chat_idempotent_stock(api_client):
    before = api_client.get("/vehicles/16").json()["stock_count"]
    payload = {"message": "reserve vehicle #16", "idempotency_key": "e2e-flow-1"}
    first = api_client.post("/api/chat", json=payload)
    second = api_client.post("/api/chat", json=payload)
    assert first.status_code == 200
    assert second.status_code == 200
    after = api_client.get("/vehicles/16").json()["stock_count"]
    assert after == before - 1
