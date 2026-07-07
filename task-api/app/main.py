from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from app import db, repository
from app.schema import Pet, Task, TaskCreate, TaskUpdate, derive_initials


@asynccontextmanager
async def lifespan(_: FastAPI):
    conn = db.connect()
    try:
        db.init_db(conn, seed=True)
    finally:
        conn.close()
    yield


app = FastAPI(title="PawDuty Task API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_conn():
    conn = db.connect()
    try:
        yield conn
    finally:
        conn.close()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/tasks", response_model=list[Task])
def list_tasks(conn=Depends(get_conn)):
    return repository.list_tasks(conn)


@app.post("/tasks", response_model=Task, status_code=201)
def create_task(payload: TaskCreate, conn=Depends(get_conn)):
    assignee = payload.assignee.model_dump()
    if assignee["name"] and not assignee["initials"]:
        assignee["initials"] = derive_initials(assignee["name"])
    task = {**payload.model_dump(), "id": uuid4().hex, "assignee": assignee}
    return repository.create_task(conn, task)


@app.get("/tasks/{task_id}", response_model=Task)
def get_task(task_id: str, conn=Depends(get_conn)):
    task = repository.get_task(conn, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.patch("/tasks/{task_id}", response_model=Task)
def update_task(task_id: str, payload: TaskUpdate, conn=Depends(get_conn)):
    fields = payload.model_dump(exclude_unset=True)
    assignee = fields.get("assignee")
    if assignee is not None and assignee.get("name") and not assignee.get("initials"):
        assignee["initials"] = derive_initials(assignee["name"])
    updated = repository.update_task(conn, task_id, fields)
    if updated is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return updated


@app.delete("/tasks/{task_id}", status_code=204)
def delete_task(task_id: str, conn=Depends(get_conn)):
    if not repository.delete_task(conn, task_id):
        raise HTTPException(status_code=404, detail="Task not found")
    return Response(status_code=204)


@app.get("/pets", response_model=list[Pet])
def list_pets(conn=Depends(get_conn)):
    return repository.list_pets(conn)
