from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytesseract

# All thermal images in ml-heat are exactly 640x480 (verified during design).
# The color-bar gradient, both calibration-number badges, and the bottom-left
# "C" unit label all live in this right-hand strip and bottom-left corner.
OVERLAY_RIGHT_X = 560
LABEL_BOTTOM_LEFT_X = 90
LABEL_BOTTOM_LEFT_Y = 440

# (x1, y1, x2, y2) - printed max/min C. Verified against pixel scans of all
# three fixtures: each badge pill spans exactly y=[30,59) (top) / y=[420,449)
# (bottom), consistently across fixtures. x1=575 sits just right of the
# lock-icon/number divider so the crop excludes the icon; x2=630 stays inside
# the pill's rounded right edge so it excludes the bright background outside
# the pill (which otherwise binarizes to a spurious digit-like blob on the
# outdoor fixture, whose background is much brighter than the indoor ones).
TOP_BADGE_BBOX = (575, 30, 630, 59)
BOTTOM_BADGE_BBOX = (575, 420, 630, 449)

MIN_BODY_FRACTION = 0.01
MAX_BODY_FRACTION = 0.9

# Below this pixel-value standard deviation (measured over the region left
# after masking overlays) an image has no discernible thermal contrast, so
# there is no way any threshold could isolate a real body from background.
# Real fixtures measure ~16-34; a uniform test/blank frame measures 0.
MIN_VALID_STD = 2.0


class NoBodyDetectedError(Exception):
    """Raised when no plausible cat body region can be isolated in a thermal image."""


def _overlay_valid_region_mask(shape: tuple[int, int]) -> np.ndarray:
    """Boolean mask, True everywhere except the overlay regions (color-bar/badge
    strip and bottom-left C-label) that mask_overlay_regions blanks out."""
    valid = np.ones(shape, dtype=bool)
    valid[:, OVERLAY_RIGHT_X:] = False
    valid[LABEL_BOTTOM_LEFT_Y:, :LABEL_BOTTOM_LEFT_X] = False
    return valid


def mask_overlay_regions(image: np.ndarray) -> np.ndarray:
    valid = _overlay_valid_region_mask(image.shape[:2])
    masked = image.copy()
    masked[~valid] = 0
    return masked


def isolate_body_mask(image: np.ndarray) -> np.ndarray:
    masked = mask_overlay_regions(image)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    valid = _overlay_valid_region_mask(gray.shape)
    valid_pixels = gray[valid]

    # Otsu's method always finds *some* split point, even in a uniform image
    # (it degenerates to thresholding at 0, marking every non-zero pixel as
    # foreground). Guard against that degenerate case explicitly: without
    # real thermal contrast there is no body to isolate.
    valid_std = float(valid_pixels.std())
    if valid_std < MIN_VALID_STD:
        raise NoBodyDetectedError(
            f"insufficient thermal contrast (std={valid_std:.2f}) to isolate a body"
        )

    # Reshape to a column vector (Nx1) rather than passing the full 2D frame:
    # cv2.threshold needs an image-shaped array, and this keeps Otsu's
    # histogram computation restricted to the valid (non-overlay) pixels only.
    thresh_val, _ = cv2.threshold(
        valid_pixels.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    mask = (gray > thresh_val) & valid

    fraction = float(mask.mean())
    if fraction < MIN_BODY_FRACTION or fraction > MAX_BODY_FRACTION:
        raise NoBodyDetectedError(
            f"body mask covers {fraction:.1%} of frame, expected between "
            f"{MIN_BODY_FRACTION:.0%} and {MAX_BODY_FRACTION:.0%}"
        )
    return mask


_BUCKET_LEVELS = 32
_QUANT_STEP = 256 // _BUCKET_LEVELS


def _build_inverse_colormap_lut() -> np.ndarray:
    """Precompute quantized-BGR -> normalized colormap position (0-1), once at import time."""
    ramp = np.arange(256, dtype=np.uint8).reshape(256, 1)
    colormap = cv2.applyColorMap(ramp, cv2.COLORMAP_INFERNO).reshape(256, 3).astype(np.int16)

    lut = np.zeros((_BUCKET_LEVELS, _BUCKET_LEVELS, _BUCKET_LEVELS), dtype=np.float32)
    centers = np.arange(_BUCKET_LEVELS) * _QUANT_STEP + _QUANT_STEP // 2
    for bi, b in enumerate(centers):
        for gi, g in enumerate(centers):
            for ri, r in enumerate(centers):
                pixel = np.array([b, g, r], dtype=np.int16)
                distances = np.sum((colormap - pixel) ** 2, axis=1)
                lut[bi, gi, ri] = np.argmin(distances) / 255.0
    return lut


_INVERSE_COLORMAP_LUT = _build_inverse_colormap_lut()


def _colors_to_normalized_positions(bgr_pixels: np.ndarray) -> np.ndarray:
    quantized = (bgr_pixels.astype(np.int32) // _QUANT_STEP).clip(0, _BUCKET_LEVELS - 1)
    return _INVERSE_COLORMAP_LUT[quantized[:, 0], quantized[:, 1], quantized[:, 2]]


def _ocr_number(image: np.ndarray, bbox: tuple[int, int, int, int]) -> float | None:
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    try:
        text = pytesseract.image_to_string(
            binary, config="--psm 7 -c tessedit_char_whitelist=0123456789"
        ).strip()
    except Exception:
        return None

    if not text.isdigit():
        return None
    return float(text)


def read_calibration_range(image: np.ndarray) -> tuple[float, float] | None:
    max_c = _ocr_number(image, TOP_BADGE_BBOX)
    min_c = _ocr_number(image, BOTTOM_BADGE_BBOX)
    if max_c is None or min_c is None or max_c <= min_c:
        return None
    return min_c, max_c


@dataclass
class TemperatureFeatures:
    average: float
    minimum: float
    maximum: float
    mode: float
    calibrated: bool


def extract_temperature_features(image_path: Path) -> TemperatureFeatures:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"could not read image: {image_path}")

    body_mask = isolate_body_mask(image)
    masked_pixels = image[body_mask]

    positions = _colors_to_normalized_positions(masked_pixels)
    calibration = read_calibration_range(image)

    if calibration is not None:
        min_c, max_c = calibration
        values = min_c + positions * (max_c - min_c)
        calibrated = True
    else:
        values = positions
        calibrated = False

    hist, bin_edges = np.histogram(values, bins=50)
    mode_index = int(np.argmax(hist))
    mode_value = float((bin_edges[mode_index] + bin_edges[mode_index + 1]) / 2)

    return TemperatureFeatures(
        average=float(values.mean()),
        minimum=float(values.min()),
        maximum=float(values.max()),
        mode=mode_value,
        calibrated=calibrated,
    )


def features_to_vector(features: TemperatureFeatures) -> np.ndarray:
    return np.array(
        [features.average, features.minimum, features.maximum, features.mode],
        dtype=np.float64,
    )
