import cv2
import numpy as np

# All thermal images in ml-heat are exactly 640x480 (verified during design).
# The color-bar gradient, both calibration-number badges, and the bottom-left
# "C" unit label all live in this right-hand strip and bottom-left corner.
OVERLAY_RIGHT_X = 560
LABEL_BOTTOM_LEFT_X = 90
LABEL_BOTTOM_LEFT_Y = 440

TOP_BADGE_BBOX = (560, 0, 640, 40)      # (x1, y1, x2, y2) - printed max C
BOTTOM_BADGE_BBOX = (560, 415, 640, 460)  # (x1, y1, x2, y2) - printed min C

MIN_BODY_FRACTION = 0.01
MAX_BODY_FRACTION = 0.9

# Below this pixel-value standard deviation (measured over the region left
# after masking overlays) an image has no discernible thermal contrast, so
# there is no way any threshold could isolate a real body from background.
# Real fixtures measure ~16-34; a uniform test/blank frame measures 0.
MIN_VALID_STD = 2.0


class NoBodyDetectedError(Exception):
    """Raised when no plausible cat body region can be isolated in a thermal image."""


def mask_overlay_regions(image: np.ndarray) -> np.ndarray:
    masked = image.copy()
    masked[:, OVERLAY_RIGHT_X:] = 0
    masked[LABEL_BOTTOM_LEFT_Y:, :LABEL_BOTTOM_LEFT_X] = 0
    return masked


def _overlay_valid_region_mask(shape: tuple) -> np.ndarray:
    """Boolean mask, True everywhere except the overlay regions blanked by mask_overlay_regions."""
    valid = np.ones(shape, dtype=bool)
    valid[:, OVERLAY_RIGHT_X:] = False
    valid[LABEL_BOTTOM_LEFT_Y:, :LABEL_BOTTOM_LEFT_X] = False
    return valid


def isolate_body_mask(image: np.ndarray) -> np.ndarray:
    masked = mask_overlay_regions(image)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    valid = _overlay_valid_region_mask(gray.shape)
    valid_pixels = gray[valid]

    # Otsu's method always finds *some* split point, even in a uniform image
    # (it degenerates to thresholding at 0, marking every non-zero pixel as
    # foreground). Guard against that degenerate case explicitly: without
    # real thermal contrast there is no body to isolate.
    if float(valid_pixels.std()) < MIN_VALID_STD:
        raise NoBodyDetectedError(
            f"insufficient thermal contrast (std={float(valid_pixels.std()):.2f}) to isolate a body"
        )

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
