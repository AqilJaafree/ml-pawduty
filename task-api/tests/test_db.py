def test_init_creates_tables(empty_conn):
    names = {
        row["name"]
        for row in empty_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {"tasks", "pets"} <= names


def test_seed_inserts_pets_and_tasks(conn):
    assert conn.execute("SELECT COUNT(*) AS c FROM pets").fetchone()["c"] == 2
    assert conn.execute("SELECT COUNT(*) AS c FROM tasks").fetchone()["c"] == 6


def test_seed_is_idempotent(conn):
    from app import db

    db.init_db(conn, seed=True)  # second call must not duplicate
    assert conn.execute("SELECT COUNT(*) AS c FROM tasks").fetchone()["c"] == 6


def test_no_seed_leaves_tables_empty(empty_conn):
    assert empty_conn.execute("SELECT COUNT(*) AS c FROM tasks").fetchone()["c"] == 0
    assert empty_conn.execute("SELECT COUNT(*) AS c FROM pets").fetchone()["c"] == 0


def test_get_db_path_honours_env(temp_db):
    from app import db

    assert db.get_db_path() == temp_db
