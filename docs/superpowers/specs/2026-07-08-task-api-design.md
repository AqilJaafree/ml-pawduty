# PawDuty Task API — Design Spec

**Date:** 2026-07-08
**Project:** `task-api` (new, at repo root — sibling to `service/` and `pawduty-fe/`)
**Scope:** Backend CRUD service for pet-care tasks, mirroring the frontend's task model

---

## 1. Overview

A standalone HTTP API providing CRUD for the tasks currently managed locally in
`pawduty-fe` (React Native / AsyncStorage). The API is a faithful mirror of the
frontend's task object shape, so it can later become a drop-in replacement for
AsyncStorage without changing the frontend's data model. It also serves a
seeded, read-only list of pets so task cards can render pet names and colors.

**Not in scope (this iteration):** wiring the frontend to call the API (frontend
keeps AsyncStorage for now); user CRUD; authentication; task pagination/filtering
(the frontend already filters client-side).

---

## 2. Stack

| Concern | Choice | Rationale |
|---|---|---|
| Language | Python 3 | Matches existing `service/` backend |
| Framework | FastAPI | Same as `service/api` |
| Validation | Pydantic v2 | Same as `service/api/schema.py` |
| Storage | SQLite via stdlib `sqlite3` | Zero-config, persists, no ORM dependency — matches the repo's minimal-deps style |
| Server | uvicorn | Same as `service/` |
| Tests | pytest + httpx `TestClient` | Same as `service/tests` |

No ORM (SQLAlchemy etc.) — a thin repository layer over `sqlite3` keeps the
dependency surface identical in spirit to `service/`.

---

## 3. Project Structure

```
task-api/
├── README.md              ← run instructions
├── requirements.txt       ← fastapi, uvicorn[standard], pydantic, pytest, httpx
├── .gitignore             ← *.db, __pycache__, .pytest_cache
├── conftest.py            ← adds package root to sys.path for pytest
├── app/
│   ├── __init__.py
│   ├── main.py            ← FastAPI app, CORS middleware, route handlers
│   ├── schema.py          ← Pydantic models (Task, TaskCreate, TaskUpdate, Pet)
│   ├── db.py              ← connection factory, schema init, first-run seed
│   └── repository.py      ← CRUD functions over sqlite (pure, testable)
└── tests/
    └── test_api.py        ← TestClient tests for every endpoint
```

**Unit boundaries:**
- `db.py` — owns the sqlite connection, table creation, and seeding. Knows the
  DB path (from `TASK_DB_PATH` env, default `task-api/pawduty_tasks.db`). Depends
  on nothing in the app.
- `repository.py` — pure CRUD functions taking a connection + arguments,
  returning plain dicts. No FastAPI imports. Independently testable.
- `schema.py` — request/response shapes and validation. No I/O.
- `main.py` — wires HTTP routes to repository calls, maps errors to status codes.

---

## 4. Data Model

### Task (API response shape — mirrors `pawduty-fe/data/seed.js`)

```jsonc
{
  "id": "string",            // server-generated uuid4 hex if omitted on create
  "title": "string",         // required, non-empty after trim
  "category": "vaccination | medicine | grooming | other",
  "petId": "string",
  "assignee": {
    "name": "string",
    "initials": "string"     // derived from name if omitted (max 2 chars, uppercase)
  },
  "date": "YYYY-MM-DD",      // required
  "time": "string",          // "" if unset
  "repeat": "once | daily | weekly | monthly",  // default "once"
  "note": "string",          // "" if unset
  "done": false              // default false
}
```

### SQLite `tasks` table

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PRIMARY KEY | uuid4 hex |
| `title` | TEXT NOT NULL | |
| `category` | TEXT NOT NULL | enum enforced at schema layer |
| `pet_id` | TEXT NOT NULL | |
| `assignee_name` | TEXT NOT NULL DEFAULT '' | |
| `assignee_initials` | TEXT NOT NULL DEFAULT '' | |
| `date` | TEXT NOT NULL | "YYYY-MM-DD" |
| `time` | TEXT NOT NULL DEFAULT '' | |
| `repeat` | TEXT NOT NULL DEFAULT 'once' | |
| `note` | TEXT NOT NULL DEFAULT '' | |
| `done` | INTEGER NOT NULL DEFAULT 0 | 0/1 ↔ bool |

`assignee` is stored as two flat columns and reconstructed into the nested object
on read (keeps rows queryable; avoids a JSON blob).

### Pet (read-only, seeded)

```jsonc
{ "id": "string", "name": "string", "species": "string", "breed": "string", "avatarColor": "string" }
```

SQLite `pets` table with matching columns (`avatar_color` → `avatarColor` on read).

### Seed data

On first init, if the `tasks` table is empty, seed the exact 6 tasks + 2 pets
from `pawduty-fe/data/seed.js` (Mochi/Buddy; medicine/vaccination/grooming/other
tasks). Task `date`s are computed as offsets from the current date at seed time
(days 0, 3, 5, 10, 15, 20) — mirroring the frontend's `daysFromToday`.

---

## 5. API Endpoints

Base: no prefix (routes mounted at root, like `service/`).

| Method | Path | Body | Success | Errors |
|---|---|---|---|---|
| `GET` | `/health` | — | `200 {"status":"ok"}` | — |
| `GET` | `/tasks` | — | `200 [Task, ...]` | — |
| `POST` | `/tasks` | `TaskCreate` | `201 Task` | `422` invalid |
| `GET` | `/tasks/{id}` | — | `200 Task` | `404` |
| `PATCH` | `/tasks/{id}` | `TaskUpdate` (all fields optional) | `200 Task` | `404`, `422` |
| `DELETE` | `/tasks/{id}` | — | `204` | `404` |
| `GET` | `/pets` | — | `200 [Pet, ...]` | — |

**Why PATCH, not PUT:** the frontend's two write actions are "add task" and
"toggle done". Partial update serves both — `{ "done": true }` is the entire
toggle payload — and lets edit-in-place send only changed fields.

### Schema layer

- `TaskCreate` — all task fields except `id` (server-generated); `title`,
  `category`, `petId`, `date` required; `assignee` optional (defaults to empty);
  enums enforced via `Literal`.
- `TaskUpdate` — every field optional; only provided fields are written.
- `Task` — full response model including `id`.
- `Pet` — read model.

Validation failures (empty `title`, invalid `category`/`repeat`, malformed body)
return `422` automatically via Pydantic. Unknown `{id}` returns `404` with a
JSON `detail` message.

---

## 6. Configuration

| Env var | Default | Purpose |
|---|---|---|
| `TASK_DB_PATH` | `task-api/pawduty_tasks.db` | SQLite file location |

CORS: `CORSMiddleware` with `allow_origins=["*"]` (dev convenience) so Expo web
and devices on the LAN can call the API.

---

## 7. Testing

pytest + `TestClient`. A fixture points `TASK_DB_PATH` at a per-test temp file
(fresh DB, seeded on init) so tests are isolated and order-independent.

Covered cases:
- `GET /health` returns ok
- Seed present: `GET /tasks` returns 6, `GET /pets` returns 2
- `POST /tasks` creates (201), server assigns `id`, initials derived from name
- `POST /tasks` with empty title → 422; invalid category → 422
- `GET /tasks/{id}` returns the created task; unknown id → 404
- `PATCH /tasks/{id}` with `{done:true}` flips done; partial edit of title works; unknown id → 404
- `DELETE /tasks/{id}` → 204, then `GET` → 404; deleting unknown id → 404

---

## 8. Success Criteria

- [ ] `task-api/` created at repo root, outside `pawduty-fe/`
- [ ] All 7 endpoints implemented and returning the documented shapes
- [ ] Task response shape byte-for-byte matches the frontend task object (nested `assignee`, camelCase `petId`)
- [ ] SQLite persists tasks across restarts; first run seeds 6 tasks + 2 pets
- [ ] All pytest tests pass
- [ ] `uvicorn app.main:app --reload` serves the API; `/docs` shows the OpenAPI UI
- [ ] README documents install + run + example curl calls
```
