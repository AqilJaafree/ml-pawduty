# Pawduty — Cat Thermal Health Worker

A FastAPI service that classifies a thermal image of a cat as **healthy** or **unhealthy**
from its body-surface temperature distribution. It reproduces (and slightly beats) the
segmentation → temperature-histogram → logistic-regression baseline from the cited
`ml-heat` paper.

- **Model:** `StandardScaler` + `LogisticRegression` over 4 temperature features
  (average, minimum, maximum, mode of the isolated body region).
- **Accuracy:** **76.5% ± 4.5%** (5-fold GroupKFold CV, grouped by cat) — beats the
  paper's 73.42% baseline.

## How it works

Each thermal frame is an INFERNO-colormapped 640×480 image with an on-image color bar
and printed min/max °C calibration badges. The pipeline:

1. **Mask overlays** — blank the right-hand color-bar strip and the bottom-left unit label.
2. **Isolate the body** — Otsu threshold on the valid region, with a contrast guard that
   raises `NoBodyDetectedError` when there's nothing to segment.
3. **Recover temperature** — invert the INFERNO colormap (precomputed LUT) to get each
   body pixel's normalized 0–1 position, then map to absolute °C using the OCR'd badge range.
4. **Extract features** — `[average, minimum, maximum, mode]` over the body's temperature histogram.
5. **Classify** — scale + logistic regression → `unhealthy` if P(sick) ≥ 0.5.

> ⚠️ **The OCR calibration step needs the system `tesseract` binary.** Without it,
> `pytesseract` fails silently and the pipeline falls back to uncalibrated 0–1 colormap
> positions, dropping accuracy to ~52% (chance). Install it with your package manager
> (e.g. `sudo pacman -S tesseract tesseract-data-eng` or `apt install tesseract-ocr`).

## Setup

```bash
cd service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You also need the system tesseract binary (see the warning above) and, for training,
the `ml-heat` dataset at the repo root (`../ml-heat`).

## Running the API

```bash
uvicorn api.main:app --reload
```

Then `POST /scan` with a thermal JPEG as multipart form field `image`:

```bash
curl -F "image=@tests/fixtures/sick_indoor_caged.jpg" http://127.0.0.1:8000/scan
```

**Response** (`200 OK`):

```json
{ "tag": "unhealthy", "confidence": 0.71 }
```

`confidence` is the probability of the predicted class (always ≥ 0.5).
If no cat body can be isolated in the image, the endpoint returns **422** with a
descriptive detail message.

Interactive docs are at `http://127.0.0.1:8000/docs`.

## Training

The shipped model lives at `models/cat_thermal_lr.joblib`. To retrain from the
`ml-heat` dataset:

```bash
python -m pawduty_ml.train
```

This cross-validates across cats (GroupKFold — a single held-out split is high-variance
on this data and can misreport the model), prints the metrics, then refits on all data
and overwrites the artifact. Reported metrics from the current model:

| Metric | Score |
|--------|-------|
| Accuracy | 0.765 |
| Precision | 0.729 |
| Recall | 0.733 |
| F1 | 0.728 |
| Specificity | 0.776 |

## Tests

```bash
pytest
```

13 tests cover the dataset walker, preprocessing (overlay masking, body isolation,
temperature extraction, OCR calibration), the model wrapper, training, and the `/scan`
endpoint. Three thermal fixtures live in `tests/fixtures/`.

## Layout

```
service/
├── api/
│   ├── main.py          # FastAPI app, POST /scan
│   └── schema.py        # ScanResponse (tag, confidence)
├── pawduty_ml/
│   ├── dataset.py       # ml-heat walker → (path, cat_id, label)
│   ├── preprocessing.py # masking, body isolation, colormap inversion, OCR, features
│   ├── model.py         # load_model / predict
│   └── train.py         # build dataset, GroupKFold eval, fit + save
├── models/
│   └── cat_thermal_lr.joblib  # {"scaler", "model"}
└── tests/
```
