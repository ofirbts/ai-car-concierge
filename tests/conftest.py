import pytest
from fastapi.testclient import TestClient

from backend import database as db


@pytest.fixture
def isolated_db(tmp_path):
    old_path = db.get_db_path()
    db_path = tmp_path / "test_inventory.db"
    db.set_db_path(db_path)
    db.init_db(force=True)
    yield db_path
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
