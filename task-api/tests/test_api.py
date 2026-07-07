def _new_task_payload(**overrides):
    payload = {
        "title": "Feed cat", "category": "medicine", "petId": "p1",
        "date": "2026-07-08", "assignee": {"name": "Alex Jaafree"},
    }
    payload.update(overrides)
    return payload


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_seed_endpoints(client):
    assert len(client.get("/tasks").json()) == 6
    assert len(client.get("/pets").json()) == 2


def test_create_returns_201_and_derives_initials(client):
    r = client.post("/tasks", json=_new_task_payload())
    assert r.status_code == 201
    body = r.json()
    assert body["id"]
    assert body["assignee"] == {"name": "Alex Jaafree", "initials": "AJ"}
    assert body["done"] is False
    assert len(client.get("/tasks").json()) == 7


def test_create_blank_title_422(client):
    r = client.post("/tasks", json=_new_task_payload(title="   "))
    assert r.status_code == 422


def test_create_bad_category_422(client):
    r = client.post("/tasks", json=_new_task_payload(category="food"))
    assert r.status_code == 422


def test_get_one_and_404(client):
    created = client.post("/tasks", json=_new_task_payload()).json()
    assert client.get(f"/tasks/{created['id']}").json()["title"] == "Feed cat"
    assert client.get("/tasks/does-not-exist").status_code == 404


def test_patch_toggle_done(client):
    created = client.post("/tasks", json=_new_task_payload()).json()
    r = client.patch(f"/tasks/{created['id']}", json={"done": True})
    assert r.status_code == 200
    assert r.json()["done"] is True


def test_patch_partial_title(client):
    created = client.post("/tasks", json=_new_task_payload()).json()
    r = client.patch(f"/tasks/{created['id']}", json={"title": "Renamed"})
    assert r.json()["title"] == "Renamed"
    assert r.json()["category"] == "medicine"  # unchanged


def test_patch_missing_404(client):
    assert client.patch("/tasks/nope", json={"done": True}).status_code == 404


def test_delete_then_404(client):
    created = client.post("/tasks", json=_new_task_payload()).json()
    assert client.delete(f"/tasks/{created['id']}").status_code == 204
    assert client.get(f"/tasks/{created['id']}").status_code == 404


def test_delete_missing_404(client):
    assert client.delete("/tasks/nope").status_code == 404
