import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    "requirements.txt",
    ".env.example",
    ".cursorrules",
    "backend/main.py",
    "backend/database.py",
    "backend/rag_service.py",
    "backend/orchestrator.py",
    "frontend/app.py",
    "data/inventory.sql",
    "data/policies/policy.md",
    "data/policies/support.md",
    "data/policies/faqs.md",
    "data/policies/maintenance.md",
    "data/policies/shipping.md",
]


def test_phase1_paths_exist():
    missing = [p for p in REQUIRED_PATHS if not (ROOT / p).is_file()]
    assert not missing, f"Missing: {missing}"


def test_inventory_sql_has_hundred_vehicles():
    sql = (ROOT / "data/inventory.sql").read_text()
    inserts = sql.count("INSERT INTO vehicles")
    assert inserts == 100


def test_policy_md_requires_2022_plus():
    policy = (ROOT / "data/policies/policy.md").read_text().lower()
    assert "2022" in policy
    assert "pending de-listing" in policy or "de-listing" in policy


def test_orchestrator_exposes_handle_chat():
    from backend import orchestrator

    assert inspect.isfunction(orchestrator.handle_chat)
    assert len(inspect.getsource(orchestrator.handle_chat).strip()) > 100
