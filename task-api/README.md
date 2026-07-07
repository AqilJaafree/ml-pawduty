# PawDuty Task API

A standalone FastAPI + SQLite CRUD service for PawDuty pet-care tasks. The JSON
shape mirrors the `pawduty-fe` frontend task model, so it can later replace the
app's AsyncStorage without changing the frontend's data model.

## Setup

```bash
cd task-api
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
cd task-api
.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

- Interactive API docs: http://localhost:8001/docs
- On first run the DB (`pawduty_tasks.db`) is created and seeded with 2 pets and
  6 demo tasks (the same seed as the app). Override the location with the
  `TASK_DB_PATH` env var.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/tasks` | List all tasks |
| POST | `/tasks` | Create a task (201) |
| GET | `/tasks/{id}` | Get one task (404 if absent) |
| PATCH | `/tasks/{id}` | Partial update — edit or toggle `done` |
| DELETE | `/tasks/{id}` | Delete (204) |
| GET | `/pets` | List seeded pets |

## Example calls

```bash
# List seeded tasks
curl http://localhost:8001/tasks

# Create a task (initials derived from the assignee name)
curl -X POST http://localhost:8001/tasks \
  -H 'Content-Type: application/json' \
  -d '{"title":"Give Mochi vitamins","category":"medicine","petId":"p1","date":"2026-07-10","assignee":{"name":"Alex Jaafree"}}'

# Toggle a task done
curl -X PATCH http://localhost:8001/tasks/<id> \
  -H 'Content-Type: application/json' -d '{"done":true}'

# Delete
curl -X DELETE http://localhost:8001/tasks/<id>
```

## Tests

```bash
cd task-api
.venv/bin/pytest -q
```

## Task shape

```jsonc
{
  "id": "string",
  "title": "string",
  "category": "vaccination | medicine | grooming | other",
  "petId": "string",
  "assignee": { "name": "string", "initials": "string" },
  "date": "YYYY-MM-DD",
  "time": "string",
  "repeat": "once | daily | weekly | monthly",
  "note": "string",
  "done": false
}
```
