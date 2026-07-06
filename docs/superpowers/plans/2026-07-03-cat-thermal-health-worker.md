# Cat Thermal Health Worker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a training pipeline that reproduces the cited paper's baseline (segmentation → temperature-histogram features → logistic regression) on the `ml-heat` dataset, and a FastAPI worker that loads the trained model and classifies an uploaded thermal image as `healthy` or `unhealthy`.

**Architecture:** A shared `pawduty_ml` package holds preprocessing (image → 4-feature temperature vector), dataset walking, training, and model load/predict logic. A thin `api` package wraps it in a single `POST /scan` FastAPI endpoint. Training and serving both call the exact same `preprocessing.py` functions to avoid train/serve skew.

**Tech Stack:** Python, OpenCV (`opencv-python-headless`), NumPy, scikit-learn, joblib, pytesseract (+ system `tesseract-ocr`), FastAPI, uvicorn, pytest, httpx.

**Spec:** `docs/superpowers/specs/2026-07-03-cat-thermal-health-worker-design.md`

---

## Prerequisite (manual, one-time, not a task)

Tesseract's binary is a system dependency of `pytesseract` and must be installed before Task 5 (OCR) will work. Install it once before starting:

```bash
sudo pacman -S tesseract tesseract-data-eng
```

Verify with `tesseract --version`.

---

## Reference facts established during design (do not re-derive)

- All 1,899 thermal images in `ml-heat` are exactly **640×480** pixels.
- Every thermal image is labeled by walking its path: `"Healthy"` or `"Sick"` appears as a path component. Filenames always end in `_Thermal_Image.jpg`. `Path("ml-heat").rglob("*_Thermal_Image.jpg")` combined with checking `"Healthy"`/`"Sick"` in `path.parts` finds all 1,899 (1,055 healthy + 844 sick) — verified during design.
- Each thermal JPG has a colorized (inferno-style) body against a near-black background, plus a baked-in overlay: a vertical color-bar gradient around x≈600–625, a "max °C" badge top-right, a "min °C" badge bottom-right, and a "°C" unit label bottom-left. The calibration range (min/max) differs per image.
- Cage bars sometimes overlap the cat's body in the dataset (e.g. `Cat_ID_93`) — the body-isolation threshold does not need to be perfect against this, matching the paper's stated method of "region masking and binary thresholding."

---

## Task 1: Project scaffolding

**Files:**
- Create: `service/requirements.txt`
- Create: `service/conftest.py`
- Create: `service/pawduty_ml/__init__.py`
- Create: `service/api/__init__.py`
- Create: `service/models/.gitkeep`

- [ ] **Step 1: Create the requirements file**

`service/requirements.txt`:
```
opencv-python-headless
numpy
scikit-learn
joblib
pytesseract
fastapi
uvicorn[standard]
python-multipart
pytest
httpx
```

- [ ] **Step 2: Create conftest.py so tests can import the packages without installation**

`service/conftest.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
```

- [ ] **Step 3: Create empty package `__init__.py` files**

`service/pawduty_ml/__init__.py`: empty file.

`service/api/__init__.py`: empty file.

- [ ] **Step 4: Create the models directory placeholder**

`service/models/.gitkeep`: empty file (keeps the directory in git before a real artifact exists).

- [ ] **Step 5: Install dependencies and verify**

```bash
cd service && pip install -r requirements.txt
```

Expected: installs without error. If `opencv-python-headless` fails to build, ensure system packages for OpenCV are present (rare on Linux with the headless wheel).

- [ ] **Step 6: Commit**

```bash
git add service/requirements.txt service/conftest.py service/pawduty_ml/__init__.py service/api/__init__.py service/models/.gitkeep
git commit -m "chore: scaffold service package structure"
```

---

## Task 2: Copy real fixture thermal images

Preprocessing tests need real thermal photos (synthetic images can't stand in for actual colorbar/badge layouts or actual cat silhouettes). Copy three known, verified-to-exist images: one indoor healthy cat, one indoor sick cat with cage bars overlapping its body (a real edge case), and one outdoor daytime healthy cat.

**Files:**
- Create: `service/tests/fixtures/healthy_indoor.jpg`
- Create: `service/tests/fixtures/sick_indoor_caged.jpg`
- Create: `service/tests/fixtures/healthy_outdoor_day.jpg`

- [ ] **Step 1: Run the copy script from the repo root**

```bash
python3 -c "
import shutil
from pathlib import Path

root = Path('ml-heat')
dest_dir = Path('service/tests/fixtures')
dest_dir.mkdir(parents=True, exist_ok=True)

targets = {
    '11.5_Thermal_Image.jpg': 'healthy_indoor.jpg',
    '93.17_Thermal_Image.jpg': 'sick_indoor_caged.jpg',
    '59.0_Thermal_Image.jpg': 'healthy_outdoor_day.jpg',
}

found = set()
for path in root.rglob('*_Thermal_Image.jpg'):
    if path.name in targets:
        shutil.copy(path, dest_dir / targets[path.name])
        found.add(path.name)

missing = set(targets) - found
if missing:
    raise SystemExit(f'missing source files: {missing}')
print('copied:', sorted(found))
"
```

Expected output: `copied: ['11.5_Thermal_Image.jpg', '59.0_Thermal_Image.jpg', '93.17_Thermal_Image.jpg']`

- [ ] **Step 2: Verify the three files exist and are non-empty**

```bash
ls -la service/tests/fixtures/
```

Expected: three `.jpg` files, each with a non-zero size.

- [ ] **Step 3: Commit**

```bash
git add service/tests/fixtures/
git commit -m "test: add real thermal image fixtures for preprocessing tests"
```

---

## Task 3: Dataset walker

**Files:**
- Create: `service/pawduty_ml/dataset.py`
- Test: `service/tests/test_dataset.py`

- [ ] **Step 1: Write the failing test**

`service/tests/test_dataset.py`:
```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd service && python -m pytest tests/test_dataset.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pawduty_ml.dataset'`

- [ ] **Step 3: Implement the dataset walker**

`service/pawduty_ml/dataset.py`:
```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd service && python -m pytest tests/test_dataset.py -v
```

Expected: PASS

- [ ] **Step 5: Verify against the real dataset**

```bash
cd service && python3 -c "
from pawduty_ml.dataset import iter_thermal_images
records = list(iter_thermal_images())
print('total:', len(records))
print('healthy:', sum(1 for r in records if r.label == 'healthy'))
print('sick:', sum(1 for r in records if r.label == 'sick'))
"
```

Expected: `total: 1899`, `healthy: 1055`, `sick: 844` (these exact counts were verified during design).

- [ ] **Step 6: Commit**

```bash
git add service/pawduty_ml/dataset.py service/tests/test_dataset.py
git commit -m "feat: add thermal image dataset walker"
```

---

## Task 4: Overlay masking and body isolation

**Files:**
- Create: `service/pawduty_ml/preprocessing.py`
- Test: `service/tests/test_preprocessing.py`

- [ ] **Step 1: Write the failing tests**

`service/tests/test_preprocessing.py`:
```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd service && python -m pytest tests/test_preprocessing.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pawduty_ml.preprocessing'`

- [ ] **Step 3: Implement overlay masking and body isolation**

`service/pawduty_ml/preprocessing.py`:
```python
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


class NoBodyDetectedError(Exception):
    """Raised when no plausible cat body region can be isolated in a thermal image."""


def mask_overlay_regions(image: np.ndarray) -> np.ndarray:
    masked = image.copy()
    masked[:, OVERLAY_RIGHT_X:] = 0
    masked[LABEL_BOTTOM_LEFT_Y:, :LABEL_BOTTOM_LEFT_X] = 0
    return masked


def isolate_body_mask(image: np.ndarray) -> np.ndarray:
    masked = mask_overlay_regions(image)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = binary > 0

    fraction = float(mask.mean())
    if fraction < MIN_BODY_FRACTION or fraction > MAX_BODY_FRACTION:
        raise NoBodyDetectedError(
            f"body mask covers {fraction:.1%} of frame, expected between "
            f"{MIN_BODY_FRACTION:.0%} and {MAX_BODY_FRACTION:.0%}"
        )
    return mask
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd service && python -m pytest tests/test_preprocessing.py -v
```

Expected: PASS (4 tests)

- [ ] **Step 5: Visually sanity-check the mask on all three fixtures**

```bash
cd service && python3 -c "
import cv2
from pathlib import Path
from pawduty_ml.preprocessing import isolate_body_mask

for name in ['healthy_indoor.jpg', 'sick_indoor_caged.jpg', 'healthy_outdoor_day.jpg']:
    image = cv2.imread(str(Path('tests/fixtures') / name))
    mask = isolate_body_mask(image)
    print(name, 'body fraction:', round(mask.mean(), 3))
"
```

Expected: all three print a fraction between 0.01 and 0.9 with no exception. If `sick_indoor_caged.jpg` raises `NoBodyDetectedError` because the cage bars fragment the mask too much, loosen `MIN_BODY_FRACTION` slightly (e.g. to `0.005`) and re-run — note the change in the commit message if you do.

- [ ] **Step 6: Commit**

```bash
git add service/pawduty_ml/preprocessing.py service/tests/test_preprocessing.py
git commit -m "feat: add overlay masking and cat body isolation"
```

---

## Task 5: Temperature calibration, colormap inversion, and feature extraction

**Files:**
- Modify: `service/pawduty_ml/preprocessing.py`
- Modify: `service/tests/test_preprocessing.py`

- [ ] **Step 1: Write the failing tests**

Append to `service/tests/test_preprocessing.py`:
```python
from pawduty_ml.preprocessing import extract_temperature_features, features_to_vector


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
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd service && python -m pytest tests/test_preprocessing.py -v
```

Expected: FAIL with `ImportError: cannot import name 'extract_temperature_features'`

- [ ] **Step 3: Implement calibration OCR, colormap inversion, and feature extraction**

Append to `service/pawduty_ml/preprocessing.py`:
```python
from dataclasses import dataclass
from pathlib import Path

import pytesseract

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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd service && python -m pytest tests/test_preprocessing.py -v
```

Expected: PASS (7 tests total in this file)

- [ ] **Step 5: Manually verify OCR calibration reads plausible values**

```bash
cd service && python3 -c "
from pathlib import Path
from pawduty_ml.preprocessing import extract_temperature_features

for name in ['healthy_indoor.jpg', 'sick_indoor_caged.jpg', 'healthy_outdoor_day.jpg']:
    f = extract_temperature_features(Path('tests/fixtures') / name)
    print(name, f)
"
```

Expected: `calibrated=True` for each, with `minimum`/`maximum` as plausible single/double-digit Celsius values (roughly 10-40). If `calibrated=False` for a fixture, the OCR badge crop coordinates (`TOP_BADGE_BBOX` / `BOTTOM_BADGE_BBOX`) may need slight adjustment — this is expected to need at most minor tuning since they were derived from visual inspection of one sample image, not pixel-verified on all three fixtures.

- [ ] **Step 6: Commit**

```bash
git add service/pawduty_ml/preprocessing.py service/tests/test_preprocessing.py
git commit -m "feat: add OCR temperature calibration and feature extraction"
```

---

## Task 6: Training — dataset builder and evaluation logic

Split into two pieces: `build_dataset` (I/O-heavy, walks real images) and `evaluate_model` (pure logic, fully unit-testable with synthetic arrays — no image I/O needed to test the metrics math).

**Files:**
- Create: `service/pawduty_ml/train.py`
- Test: `service/tests/test_train.py`

- [ ] **Step 1: Write the failing test**

`service/tests/test_train.py`:
```python
import numpy as np

from pawduty_ml.train import evaluate_model


def test_evaluate_model_reports_expected_metric_keys() -> None:
    # 4 groups (cats), perfectly separable by the first feature, so the split
    # and metrics are deterministic regardless of which group lands in test.
    X = np.array(
        [
            [10.0, 8.0, 12.0, 10.0],
            [11.0, 9.0, 13.0, 11.0],
            [30.0, 28.0, 33.0, 30.0],
            [31.0, 29.0, 34.0, 31.0],
        ]
    )
    y = np.array([0, 0, 1, 1])
    groups = np.array(["cat_a", "cat_b", "cat_c", "cat_d"])

    report = evaluate_model(X, y, groups, test_size=0.5, random_state=0)

    assert set(report) == {"accuracy", "precision", "recall", "f1", "specificity"}
    for value in report.values():
        assert 0.0 <= value <= 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd service && python -m pytest tests/test_train.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pawduty_ml.train'`

- [ ] **Step 3: Implement the dataset builder and evaluation logic**

`service/pawduty_ml/train.py`:
```python
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from pawduty_ml.dataset import DEFAULT_DATASET_ROOT, iter_thermal_images
from pawduty_ml.preprocessing import NoBodyDetectedError, extract_temperature_features, features_to_vector

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "cat_thermal_lr.joblib"


def build_dataset(dataset_root: Path = DEFAULT_DATASET_ROOT) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []

    for record in iter_thermal_images(dataset_root):
        try:
            temp_features = extract_temperature_features(record.path)
        except NoBodyDetectedError as exc:
            logger.warning("skipping %s: %s", record.path, exc)
            continue
        features.append(features_to_vector(temp_features))
        labels.append(1 if record.label == "sick" else 0)
        groups.append(record.cat_id)

    return np.array(features), np.array(labels), np.array(groups)


def evaluate_model(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, test_size: float = 0.2, random_state: int = 42
) -> dict[str, float]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    y_train, y_test = y[train_idx], y[test_idx]

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "specificity": specificity,
    }


def train_and_save(dataset_root: Path = DEFAULT_DATASET_ROOT, model_path: Path = MODEL_PATH) -> dict[str, float]:
    X, y, groups = build_dataset(dataset_root)
    report = evaluate_model(X, y, groups)

    # Refit on all data for the artifact that actually gets served.
    scaler = StandardScaler().fit(X)
    model = LogisticRegression().fit(scaler.transform(X), y)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "model": model}, model_path)

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = train_and_save()
    for key, value in result.items():
        print(f"{key}: {value:.4f}")
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd service && python -m pytest tests/test_train.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add service/pawduty_ml/train.py service/tests/test_train.py
git commit -m "feat: add training dataset builder and evaluation logic"
```

---

## Task 7: Model load/predict wrapper

**Files:**
- Create: `service/pawduty_ml/model.py`
- Test: `service/tests/test_model.py`

- [ ] **Step 1: Write the failing test**

`service/tests/test_model.py`:
```python
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pawduty_ml.model import load_model, predict


def _write_fake_artifact(path: Path) -> None:
    X = np.array([[20.0, 18.0, 22.0, 20.0], [30.0, 28.0, 33.0, 31.0]])
    y = np.array([0, 1])
    scaler = StandardScaler().fit(X)
    model = LogisticRegression().fit(scaler.transform(X), y)
    joblib.dump({"scaler": scaler, "model": model}, path)


def test_predict_returns_healthy_for_cool_features(tmp_path: Path) -> None:
    artifact_path = tmp_path / "model.joblib"
    _write_fake_artifact(artifact_path)
    scaler, model = load_model(artifact_path)

    result = predict(np.array([20.0, 18.0, 22.0, 20.0]), scaler, model)

    assert result.tag == "healthy"
    assert 0.5 <= result.confidence <= 1.0


def test_predict_returns_unhealthy_for_warm_features(tmp_path: Path) -> None:
    artifact_path = tmp_path / "model.joblib"
    _write_fake_artifact(artifact_path)
    scaler, model = load_model(artifact_path)

    result = predict(np.array([30.0, 28.0, 33.0, 31.0]), scaler, model)

    assert result.tag == "unhealthy"
    assert 0.5 <= result.confidence <= 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd service && python -m pytest tests/test_model.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pawduty_ml.model'`

- [ ] **Step 3: Implement the model wrapper**

`service/pawduty_ml/model.py`:
```python
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "cat_thermal_lr.joblib"


@dataclass
class Prediction:
    tag: str
    confidence: float


def load_model(model_path: Path = MODEL_PATH):
    artifact = joblib.load(model_path)
    return artifact["scaler"], artifact["model"]


def predict(feature_vector: np.ndarray, scaler, model) -> Prediction:
    scaled = scaler.transform(feature_vector.reshape(1, -1))
    probability_sick = float(model.predict_proba(scaled)[0, 1])

    if probability_sick >= 0.5:
        return Prediction(tag="unhealthy", confidence=probability_sick)
    return Prediction(tag="healthy", confidence=1.0 - probability_sick)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd service && python -m pytest tests/test_model.py -v
```

Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add service/pawduty_ml/model.py service/tests/test_model.py
git commit -m "feat: add model load and predict wrapper"
```

---

## Task 8: FastAPI /scan endpoint

Uses FastAPI's dependency-override mechanism so the API test doesn't depend on a real trained artifact existing on disk yet.

**Files:**
- Create: `service/api/schema.py`
- Create: `service/api/main.py`
- Test: `service/tests/test_api.py`

- [ ] **Step 1: Write the failing test**

`service/tests/test_api.py`:
```python
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from api.main import app, get_model

FIXTURES = Path(__file__).parent / "fixtures"


def _fake_model_bundle():
    X = np.array(
        [[20.0, 18.0, 22.0, 20.0], [30.0, 28.0, 33.0, 31.0], [21.0, 19.0, 23.0, 21.0], [29.0, 27.0, 32.0, 30.0]]
    )
    y = np.array([0, 1, 0, 1])
    scaler = StandardScaler().fit(X)
    model = LogisticRegression().fit(scaler.transform(X), y)
    return scaler, model


app.dependency_overrides[get_model] = _fake_model_bundle
client = TestClient(app)


def test_scan_returns_a_tag_and_confidence_for_a_real_image() -> None:
    with open(FIXTURES / "healthy_indoor.jpg", "rb") as image_file:
        response = client.post("/scan", files={"image": ("healthy_indoor.jpg", image_file, "image/jpeg")})

    assert response.status_code == 200
    body = response.json()
    assert body["tag"] in ("healthy", "unhealthy")
    assert 0.0 <= body["confidence"] <= 1.0


def test_scan_returns_422_for_an_image_with_no_detectable_body() -> None:
    import io

    import cv2
    import numpy as np

    blank = np.full((480, 640, 3), 20, dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", blank)
    assert success
    blank_file = io.BytesIO(encoded.tobytes())

    response = client.post("/scan", files={"image": ("blank.jpg", blank_file, "image/jpeg")})

    assert response.status_code == 422
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd service && python -m pytest tests/test_api.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'api.main'`

- [ ] **Step 3: Implement the response schema**

`service/api/schema.py`:
```python
from typing import Literal

from pydantic import BaseModel


class ScanResponse(BaseModel):
    tag: Literal["healthy", "unhealthy"]
    confidence: float
```

- [ ] **Step 4: Implement the FastAPI app**

`service/api/main.py`:
```python
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from api.schema import ScanResponse
from pawduty_ml.model import load_model, predict
from pawduty_ml.preprocessing import NoBodyDetectedError, extract_temperature_features, features_to_vector

app = FastAPI(title="Pawduty Cat Thermal Health Worker")


def get_model():
    return load_model()


@app.post("/scan", response_model=ScanResponse)
async def scan(image: UploadFile = File(...), model_bundle=Depends(get_model)) -> ScanResponse:
    scaler, model = model_bundle
    contents = await image.read()

    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        tmp.write(contents)
        tmp.flush()
        try:
            features = extract_temperature_features(Path(tmp.name))
        except NoBodyDetectedError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    vector = features_to_vector(features)
    result = predict(vector, scaler, model)
    return ScanResponse(tag=result.tag, confidence=result.confidence)
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd service && python -m pytest tests/test_api.py -v
```

Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add service/api/schema.py service/api/main.py service/tests/test_api.py
git commit -m "feat: add POST /scan FastAPI endpoint"
```

---

## Task 9: Train on the real dataset and ship the artifact

Everything up to this point has been testable without a real trained model. Now run the actual training pipeline against all 1,899 thermal images and commit the resulting artifact so the API has something real to load.

**Files:**
- Create: `service/models/cat_thermal_lr.joblib` (generated, not hand-written)

- [ ] **Step 1: Run the full training pipeline**

```bash
cd service && python -m pawduty_ml.train
```

Expected: prints `accuracy`, `precision`, `recall`, `f1`, `specificity` as floats between 0 and 1, and creates `service/models/cat_thermal_lr.joblib`. This should complete in well under a minute — 1,899 images through OpenCV + OCR is not compute-heavy. Some images may log a `skipping ...: body mask covers ...` warning (rejected by `isolate_body_mask`) — that's expected and fine as long as most images are used.

- [ ] **Step 2: Sanity-check the reported metrics against the paper's baseline**

The cited paper reports 73.42% accuracy on its own train/test split. Our split is by Cat ID (stricter, see spec) so an exact match isn't expected, but accuracy in a roughly similar range (well above 50% chance level) confirms the pipeline is learning signal rather than noise. If accuracy is at or near 50%, treat that as a bug signal and re-check the feature extraction (Task 5) before proceeding.

- [ ] **Step 3: Run the full test suite once more end-to-end**

```bash
cd service && python -m pytest -v
```

Expected: all tests across `test_dataset.py`, `test_preprocessing.py`, `test_train.py`, `test_model.py`, `test_api.py` PASS.

- [ ] **Step 4: Start the worker locally and verify a real request**

```bash
cd service && uvicorn api.main:app --port 8000 &
sleep 2
curl -s -X POST http://127.0.0.1:8000/scan -F "image=@tests/fixtures/healthy_indoor.jpg"
kill %1
```

Expected: a JSON response like `{"tag":"healthy","confidence":0.87}` (exact numbers will vary), not an error.

- [ ] **Step 5: Commit the trained artifact**

```bash
git add service/models/cat_thermal_lr.joblib
git commit -m "feat: train and ship cat thermal health baseline model"
```

---

## Self-review notes

- **Spec coverage:** dataset walking (Task 3), preprocessing/segmentation (Task 4), OCR temperature calibration + histogram features (Task 5), cat-ID split + logistic regression + paper-matching metrics (Task 6), model artifact load/predict (Task 7), `POST /scan` with 422 on no-body-detected (Task 8), full training run + manual worker verification (Task 9). All spec sections are covered.
- **Type/signature consistency:** `features_to_vector` order (average, minimum, maximum, mode) is defined once in Task 5 and consumed identically in Task 6 (`build_dataset`) and Task 8 (`api/main.py`) — no divergent copies. `NoBodyDetectedError` is defined once in Task 4 and imported (not redefined) everywhere else. `Prediction.tag`/`Prediction.confidence` field names match `ScanResponse.tag`/`ScanResponse.confidence` in Task 8.
- **No placeholders:** every task shows complete, runnable code; the only intentionally-approximate values are the overlay bounding box pixel constants (Task 4) and OCR badge bounding boxes (Task 5), which are flagged explicitly as visually-derived and given a concrete verification step plus a concrete fallback instruction if they need adjustment — this is a known tuning point, not an unfinished placeholder.
