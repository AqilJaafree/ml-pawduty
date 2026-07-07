import os
import sqlite3
from datetime import date, timedelta
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "pawduty_tasks.db"

SEED_PETS = [
    {"id": "p1", "name": "Mochi", "species": "cat", "breed": "Scottish Fold", "avatar_color": "#FFD166"},
    {"id": "p2", "name": "Buddy", "species": "dog", "breed": "Golden Retriever", "avatar_color": "#FF8C42"},
]


def _days_from_today(n: int) -> str:
    return (date.today() + timedelta(days=n)).isoformat()


def seed_tasks() -> list[dict]:
    """The 6 demo tasks from pawduty-fe/data/seed.js, dated relative to today."""
    return [
        {"id": "t1", "title": "Give Mochi her flea medicine", "category": "medicine",
         "pet_id": "p1", "assignee_name": "Alex", "assignee_initials": "A",
         "date": _days_from_today(0), "time": "08:00", "repeat": "daily", "note": "", "done": 0},
        {"id": "t2", "title": "Buddy's rabies booster shot", "category": "vaccination",
         "pet_id": "p2", "assignee_name": "Alex", "assignee_initials": "A",
         "date": _days_from_today(3), "time": "", "repeat": "once", "note": "Check with Dr. Smith", "done": 0},
        {"id": "t3", "title": "Trim Mochi's nails", "category": "grooming",
         "pet_id": "p1", "assignee_name": "Alex", "assignee_initials": "A",
         "date": _days_from_today(5), "time": "14:00", "repeat": "weekly", "note": "", "done": 0},
        {"id": "t4", "title": "Monthly heartworm pill for Buddy", "category": "medicine",
         "pet_id": "p2", "assignee_name": "Alex", "assignee_initials": "A",
         "date": _days_from_today(10), "time": "", "repeat": "monthly", "note": "", "done": 0},
        {"id": "t5", "title": "Vet checkup for Buddy", "category": "other",
         "pet_id": "p2", "assignee_name": "Alex", "assignee_initials": "A",
         "date": _days_from_today(15), "time": "10:00", "repeat": "once", "note": "Annual checkup", "done": 0},
        {"id": "t6", "title": "Mochi's deworming tablet", "category": "medicine",
         "pet_id": "p1", "assignee_name": "Alex", "assignee_initials": "A",
         "date": _days_from_today(20), "time": "", "repeat": "monthly", "note": "", "done": 0},
    ]


def get_db_path() -> str:
    return os.environ.get("TASK_DB_PATH", str(DEFAULT_DB_PATH))


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection, seed: bool = True) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS pets (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            species TEXT NOT NULL,
            breed TEXT NOT NULL,
            avatar_color TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            category TEXT NOT NULL,
            pet_id TEXT NOT NULL,
            assignee_name TEXT NOT NULL DEFAULT '',
            assignee_initials TEXT NOT NULL DEFAULT '',
            date TEXT NOT NULL,
            time TEXT NOT NULL DEFAULT '',
            repeat TEXT NOT NULL DEFAULT 'once',
            note TEXT NOT NULL DEFAULT '',
            done INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    conn.commit()

    if not seed:
        return
    already = conn.execute("SELECT COUNT(*) AS c FROM tasks").fetchone()["c"]
    if already:
        return

    conn.executemany(
        "INSERT INTO pets (id, name, species, breed, avatar_color) VALUES "
        "(:id, :name, :species, :breed, :avatar_color)",
        SEED_PETS,
    )
    conn.executemany(
        "INSERT INTO tasks (id, title, category, pet_id, assignee_name, assignee_initials, "
        "date, time, repeat, note, done) VALUES "
        "(:id, :title, :category, :pet_id, :assignee_name, :assignee_initials, "
        ":date, :time, :repeat, :note, :done)",
        seed_tasks(),
    )
    conn.commit()
