import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Point the API at a fresh per-test SQLite file."""
    path = tmp_path / "test.db"
    monkeypatch.setenv("TASK_DB_PATH", str(path))
    return str(path)


@pytest.fixture
def conn(temp_db):
    """A connection to a freshly-initialised, seeded DB."""
    from app import db

    connection = db.connect()
    db.init_db(connection, seed=True)
    yield connection
    connection.close()


@pytest.fixture
def empty_conn(temp_db):
    """A connection to an initialised but UNSEEDED DB."""
    from app import db

    connection = db.connect()
    db.init_db(connection, seed=False)
    yield connection
    connection.close()


@pytest.fixture
def client(temp_db):
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
