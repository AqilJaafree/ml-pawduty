from app import repository


def _sample_task(task_id="new1"):
    return {
        "id": task_id, "title": "Feed cat", "category": "medicine", "petId": "p1",
        "assignee": {"name": "Alex", "initials": "A"},
        "date": "2026-07-08", "time": "09:00", "repeat": "once", "note": "n", "done": False,
    }


def test_list_pets_returns_seeded_shape(conn):
    pets = repository.list_pets(conn)
    assert len(pets) == 2
    assert pets[0]["avatarColor"].startswith("#")
    assert set(pets[0].keys()) == {"id", "name", "species", "breed", "avatarColor"}


def test_list_tasks_returns_seeded(conn):
    tasks = repository.list_tasks(conn)
    assert len(tasks) == 6
    t = tasks[0]
    assert "petId" in t and "pet_id" not in t
    assert set(t["assignee"].keys()) == {"name", "initials"}
    assert isinstance(t["done"], bool)


def test_create_then_get(empty_conn):
    created = repository.create_task(empty_conn, _sample_task())
    assert created["id"] == "new1"
    assert created["assignee"]["initials"] == "A"
    got = repository.get_task(empty_conn, "new1")
    assert got["title"] == "Feed cat"


def test_get_missing_returns_none(empty_conn):
    assert repository.get_task(empty_conn, "nope") is None


def test_update_partial_toggle_done(empty_conn):
    repository.create_task(empty_conn, _sample_task())
    updated = repository.update_task(empty_conn, "new1", {"done": True})
    assert updated["done"] is True
    assert updated["title"] == "Feed cat"  # untouched


def test_update_title_and_assignee(empty_conn):
    repository.create_task(empty_conn, _sample_task())
    updated = repository.update_task(
        empty_conn, "new1",
        {"title": "New title", "assignee": {"name": "Bob", "initials": "B"}},
    )
    assert updated["title"] == "New title"
    assert updated["assignee"] == {"name": "Bob", "initials": "B"}


def test_update_missing_returns_none(empty_conn):
    assert repository.update_task(empty_conn, "nope", {"done": True}) is None


def test_delete(empty_conn):
    repository.create_task(empty_conn, _sample_task())
    assert repository.delete_task(empty_conn, "new1") is True
    assert repository.get_task(empty_conn, "new1") is None


def test_delete_missing_returns_false(empty_conn):
    assert repository.delete_task(empty_conn, "nope") is False
