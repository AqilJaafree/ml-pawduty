import sqlite3


def _row_to_task(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "title": row["title"],
        "category": row["category"],
        "petId": row["pet_id"],
        "assignee": {"name": row["assignee_name"], "initials": row["assignee_initials"]},
        "date": row["date"],
        "time": row["time"],
        "repeat": row["repeat"],
        "note": row["note"],
        "done": bool(row["done"]),
    }


def _row_to_pet(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "species": row["species"],
        "breed": row["breed"],
        "avatarColor": row["avatar_color"],
    }


def list_tasks(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM tasks ORDER BY date, id").fetchall()
    return [_row_to_task(r) for r in rows]


def get_task(conn: sqlite3.Connection, task_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return _row_to_task(row) if row else None


def create_task(conn: sqlite3.Connection, task: dict) -> dict:
    conn.execute(
        "INSERT INTO tasks (id, title, category, pet_id, assignee_name, assignee_initials, "
        "date, time, repeat, note, done) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            task["id"], task["title"], task["category"], task["petId"],
            task["assignee"]["name"], task["assignee"]["initials"],
            task["date"], task["time"], task["repeat"], task["note"], int(task["done"]),
        ),
    )
    conn.commit()
    return get_task(conn, task["id"])


# Maps API field name -> SQL column for scalar fields.
_SCALAR_COLUMNS = {
    "title": "title",
    "category": "category",
    "petId": "pet_id",
    "date": "date",
    "time": "time",
    "repeat": "repeat",
    "note": "note",
}


def update_task(conn: sqlite3.Connection, task_id: str, fields: dict) -> dict | None:
    if get_task(conn, task_id) is None:
        return None

    sets: list[str] = []
    values: list = []
    for key, column in _SCALAR_COLUMNS.items():
        if key in fields:
            sets.append(f"{column} = ?")
            values.append(fields[key])
    if "done" in fields:
        sets.append("done = ?")
        values.append(int(fields["done"]))
    if fields.get("assignee") is not None:
        sets.append("assignee_name = ?")
        values.append(fields["assignee"]["name"])
        sets.append("assignee_initials = ?")
        values.append(fields["assignee"]["initials"])

    if sets:
        values.append(task_id)
        conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?", values)
        conn.commit()
    return get_task(conn, task_id)


def delete_task(conn: sqlite3.Connection, task_id: str) -> bool:
    cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    return cur.rowcount > 0


def list_pets(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM pets ORDER BY id").fetchall()
    return [_row_to_pet(r) for r in rows]
