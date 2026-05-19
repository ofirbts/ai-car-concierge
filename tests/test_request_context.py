def test_request_id_header(api_client):
    response = api_client.get("/health", headers={"X-Request-ID": "test-123"})
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == "test-123"
