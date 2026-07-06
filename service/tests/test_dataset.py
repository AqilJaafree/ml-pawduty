from pathlib import Path

from pawduty_ml.dataset import iter_thermal_images


def _make_thermal_image(root: Path, *parts: str, filename: str) -> None:
    directory = root.joinpath(*parts)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_bytes(b"not a real jpg, path parsing does not read content")


def test_iter_thermal_images_labels_by_path_and_extracts_cat_id(tmp_path: Path) -> None:
    _make_thermal_image(
        tmp_path, "Controlled_environment_(Indoor)", "Healthy", "Thermal", "Cat_ID_11",
        filename="11.5_Thermal_Image.jpg",
    )
    _make_thermal_image(
        tmp_path, "Controlled_environment_(Indoor)", "Sick", "Thermal", "Cat_ID_93",
        filename="93.17_Thermal_Image.jpg",
    )
    _make_thermal_image(
        tmp_path, "Controlled_environment_(Indoor)", "Healthy", "Digital", "Cat_ID_11",
        filename="11.5_Digital_Image.jpg",
    )

    records = sorted(iter_thermal_images(tmp_path), key=lambda r: r.cat_id)

    assert len(records) == 2
    assert records[0].cat_id == "Cat_ID_11"
    assert records[0].label == "healthy"
    assert records[1].cat_id == "Cat_ID_93"
    assert records[1].label == "sick"
