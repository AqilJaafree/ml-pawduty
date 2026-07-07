import pytest
from pydantic import ValidationError

from app.schema import Assignee, Pet, Task, TaskCreate, TaskUpdate, derive_initials


def test_derive_initials_single_name():
    assert derive_initials("Alex") == "A"


def test_derive_initials_two_words_uppercased():
    assert derive_initials("alex jaafree") == "AJ"


def test_derive_initials_caps_at_two():
    assert derive_initials("Mary Jane Watson") == "MJ"


def test_derive_initials_empty():
    assert derive_initials("") == ""


def test_task_create_minimal_valid():
    tc = TaskCreate(title="Feed cat", category="medicine", petId="p1", date="2026-07-08")
    assert tc.repeat == "once"
    assert tc.done is False
    assert tc.assignee.name == ""


def test_task_create_rejects_blank_title():
    with pytest.raises(ValidationError):
        TaskCreate(title="   ", category="medicine", petId="p1", date="2026-07-08")


def test_task_create_rejects_bad_category():
    with pytest.raises(ValidationError):
        TaskCreate(title="x", category="food", petId="p1", date="2026-07-08")


def test_task_update_rejects_blank_title():
    with pytest.raises(ValidationError):
        TaskUpdate(title="   ")


def test_derive_initials_multiple_spaces():
    assert derive_initials("John  Doe") == "JD"


def test_derive_initials_leading_trailing_spaces():
    assert derive_initials("  Alex Smith  ") == "AS"


def test_task_update_all_optional():
    tu = TaskUpdate(done=True)
    dumped = tu.model_dump(exclude_unset=True)
    assert dumped == {"done": True}


def test_task_and_pet_roundtrip():
    task = Task(
        id="t1", title="x", category="other", petId="p1",
        assignee=Assignee(name="Alex", initials="A"),
        date="2026-07-08", time="", repeat="once", note="", done=False,
    )
    assert task.assignee.initials == "A"
    pet = Pet(id="p1", name="Mochi", species="cat", breed="Scottish Fold", avatarColor="#FFD166")
    assert pet.avatarColor == "#FFD166"
