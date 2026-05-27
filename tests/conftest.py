import os
from pathlib import Path

if os.environ.get("REQUIRE_PRODUCTION_API_KEY") != "true":
    os.environ["API_KEY"] = ""
os.environ.setdefault("GOOGLE_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ.setdefault("RESEND_API_KEY", "")
os.environ.setdefault("CHAT_GOVERNOR_JOURNAL", "/tmp/ai_car_concierge_chat_governor_test.jsonl")
os.environ.setdefault("ENABLE_EXPERIMENTAL", "true")

import pytest
from fastapi.testclient import TestClient

from backend import database as db
from backend.config import reset_settings_cache
from backend.intent import _inventory_makes


def _clear_inventory_makes_cache() -> None:
    _inventory_makes.cache_clear()


@pytest.fixture(autouse=True)
def _reset_settings_each_test():
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    from limits.storage import MemoryStorage
    from backend.main import limiter

    limiter._storage = MemoryStorage()
    limiter.reset()
    yield
    limiter._storage = MemoryStorage()
    limiter.reset()


@pytest.fixture(autouse=True)
def _clean_chat_governor_journal():
    journal_path = Path(os.environ["CHAT_GOVERNOR_JOURNAL"])
    if journal_path.exists():
        journal_path.unlink()
    yield
    if journal_path.exists():
        journal_path.unlink()


@pytest.fixture
def isolated_db(tmp_path):
    old_path = db.get_db_path()
    db_path = tmp_path / "test_inventory.db"
    db.set_db_path(db_path)
    db.init_db(force=True)
    _clear_inventory_makes_cache()
    yield db_path
    hash_path = db_path.with_suffix(db_path.suffix + ".sqlhash")
    if hash_path.exists():
        hash_path.unlink()
    if db_path.exists():
        db_path.unlink()
    db.set_db_path(old_path)


@pytest.fixture(autouse=True)
def _init_inventory_for_tests(request, tmp_path):
    if "isolated_db" in request.fixturenames:
        yield
        return
    old_path = db.get_db_path()
    db_path = tmp_path / "autouse_inventory.db"
    db.set_db_path(db_path)
    db.init_db(force=True)
    _clear_inventory_makes_cache()
    yield
    hash_path = db_path.with_suffix(db_path.suffix + ".sqlhash")
    if hash_path.exists():
        hash_path.unlink()
    if db_path.exists():
        db_path.unlink()
    db.set_db_path(old_path)


@pytest.fixture
def api_client(isolated_db):
    from backend.main import app

    with TestClient(app) as client:
        yield client
