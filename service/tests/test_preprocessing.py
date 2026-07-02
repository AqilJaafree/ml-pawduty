from pathlib import Path

import cv2
import numpy as np
import pytest

from pawduty_ml.preprocessing import (
    NoBodyDetectedError,
    extract_temperature_features,
    features_to_vector,
    isolate_body_mask,
    mask_overlay_regions,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_mask_overlay_regions_blanks_right_column_and_bottom_left_label() -> None:
    image = np.full((480, 640, 3), 100, dtype=np.uint8)

    masked = mask_overlay_regions(image)

    assert masked[10, 600].tolist() == [0, 0, 0]
    assert masked[450, 10].tolist() == [0, 0, 0]
    assert masked[240, 300].tolist() == [100, 100, 100]


def test_isolate_body_mask_finds_a_real_cat_body() -> None:
    image = cv2.imread(str(FIXTURES / "healthy_indoor.jpg"))

    mask = isolate_body_mask(image)

    fraction = mask.mean()
    assert 0.01 < fraction < 0.9


def test_isolate_body_mask_rejects_uniformly_dark_background() -> None:
    image = np.full((480, 640, 3), 20, dtype=np.uint8)

    with pytest.raises(NoBodyDetectedError):
        isolate_body_mask(image)


def test_isolate_body_mask_rejects_uniformly_bright_image() -> None:
    image = np.full((480, 640, 3), 200, dtype=np.uint8)

    with pytest.raises(NoBodyDetectedError):
        isolate_body_mask(image)


def test_extract_temperature_features_on_real_indoor_fixture() -> None:
    features = extract_temperature_features(FIXTURES / "healthy_indoor.jpg")

    assert features.minimum <= features.average <= features.maximum
    assert features.minimum <= features.mode <= features.maximum
    # Cat thermal images in this dataset range roughly 10-40 C; a wide sanity band,
    # not a precision check (OCR/colormap accuracy is validated visually, not asserted).
    assert -10 < features.minimum
    assert features.maximum < 60


def test_extract_temperature_features_falls_back_when_ocr_is_unavailable(monkeypatch) -> None:
    import pytesseract

    def _broken_ocr(*args, **kwargs):
        raise RuntimeError("tesseract is not installed")

    monkeypatch.setattr(pytesseract, "image_to_string", _broken_ocr)

    features = extract_temperature_features(FIXTURES / "healthy_indoor.jpg")

    assert features.calibrated is False
    assert 0.0 <= features.minimum <= features.maximum <= 1.0


def test_features_to_vector_matches_dataclass_order() -> None:
    features = extract_temperature_features(FIXTURES / "healthy_outdoor_day.jpg")

    vector = features_to_vector(features)

    assert vector.shape == (4,)
    assert vector[0] == features.average
    assert vector[1] == features.minimum
    assert vector[2] == features.maximum
    assert vector[3] == features.mode
