from pathlib import Path

import cv2
import numpy as np
import pytest

from pawduty_ml.preprocessing import NoBodyDetectedError, isolate_body_mask, mask_overlay_regions

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
