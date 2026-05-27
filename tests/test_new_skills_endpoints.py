import time
from pathlib import Path


def test_codebase_packager_endpoint(api_client, tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("x = 1\n", encoding="utf-8")
    payload = {"root_path": str(repo)}
    response = api_client.post("/skills/codebase-packager", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["files"]
    assert data["idempotency_key"].startswith("codebase_packager:")


def test_job_submit_and_status_endpoint(api_client, tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("print('x')\n", encoding="utf-8")
    payload = {"kind": "codebase_packager", "payload": {"root_path": str(repo)}}
    submit = api_client.post("/jobs/submit", json=payload)
    assert submit.status_code == 200
    job_id = submit.json()["job_id"]

    deadline = time.time() + 2.0
    state = "pending"
    while time.time() < deadline:
        status = api_client.get(f"/jobs/{job_id}")
        assert status.status_code == 200
        state = status.json()["state"]
        if state in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.02)

    assert state == "completed"


def test_iteration_governor_endpoint(api_client):
    payload = {"run_id": "api-run-1", "initial_state": {"start": 1}, "steps": ["s1", "s2"]}
    response = api_client.post("/governor/iteration/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["last_event"] == "RUN_COMPLETED"
    assert data["state"]["last_step"] == "s2"
