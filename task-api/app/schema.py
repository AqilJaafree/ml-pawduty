from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

Category = Literal["vaccination", "medicine", "grooming", "other"]
Repeat = Literal["once", "daily", "weekly", "monthly"]


def derive_initials(name: str) -> str:
    """First letter of each whitespace-separated word, first two, uppercased.

    Mirrors the frontend: name.split(' ').map(p => p[0]).join('').slice(0, 2).toUpperCase()
    """
    return "".join(part[0] for part in name.split())[:2].upper()


def _validate_title(v):
    if v is None:
        return v
    stripped = v.strip()
    if not stripped:
        raise ValueError("title must not be blank")
    return stripped


class Assignee(BaseModel):
    name: str = ""
    initials: str = ""


class TaskCreate(BaseModel):
    title: str
    category: Category
    petId: str
    date: str
    assignee: Assignee = Field(default_factory=Assignee)
    time: str = ""
    repeat: Repeat = "once"
    note: str = ""
    done: bool = False

    @field_validator("title")
    @classmethod
    def _title_not_blank(cls, v: str) -> str:
        return _validate_title(v)


class TaskUpdate(BaseModel):
    title: Optional[str] = None
    category: Optional[Category] = None
    petId: Optional[str] = None
    date: Optional[str] = None
    assignee: Optional[Assignee] = None
    time: Optional[str] = None
    repeat: Optional[Repeat] = None
    note: Optional[str] = None
    done: Optional[bool] = None

    @field_validator("title")
    @classmethod
    def _title_not_blank(cls, v):
        return _validate_title(v)


class Task(BaseModel):
    id: str
    title: str
    category: Category
    petId: str
    assignee: Assignee
    date: str
    time: str
    repeat: Repeat
    note: str
    done: bool


class Pet(BaseModel):
    id: str
    name: str
    species: str
    breed: str
    avatarColor: str
