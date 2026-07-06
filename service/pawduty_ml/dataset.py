from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

Label = Literal["healthy", "sick"]

DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[2] / "ml-heat"


@dataclass
class ThermalImageRecord:
    path: Path
    cat_id: str
    label: Label


def iter_thermal_images(dataset_root: Path = DEFAULT_DATASET_ROOT) -> Iterator[ThermalImageRecord]:
    for path in dataset_root.rglob("*_Thermal_Image.jpg"):
        parts = path.parts
        if "Healthy" in parts:
            label: Label = "healthy"
        elif "Sick" in parts:
            label = "sick"
        else:
            continue
        yield ThermalImageRecord(path=path, cat_id=path.parent.name, label=label)
