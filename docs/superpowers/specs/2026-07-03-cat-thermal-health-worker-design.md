# Cat Thermal Health Worker — Design

## Goal

Build a worker that classifies a cat as `healthy` or `unhealthy` from a thermal image, using the technique described in ["A Thermal Imaging Dataset and Baseline for Cat Health Monitoring"](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1650223/full) (segmentation → temperature-histogram features → logistic regression), trained on the `ml-heat` dataset already in this repo.

## Scope

**In scope:**
- An offline training pipeline that reproduces the paper's baseline technique on `ml-heat` and produces a trained model artifact.
- A FastAPI inference service ("the worker") that loads that artifact and classifies an uploaded thermal image via `POST /scan`.

**Out of scope (future work, separate specs):**
- Client-side capture app (mobile/web UI that takes the photo and calls the worker).
- Deployment/hosting configuration.
- CNN-based upgrade path (paper notes this as future work; not built here).

## Dataset

`ml-heat/Thermal Image Dataset of cats for their health monitoring and_classification/.../` contains 2,426 images across 94 cats:

- **Thermal** images (1,899) — colorized (ironbow-style) thermal photos. Each has a baked-in color-bar overlay on the right edge with two printed calibration numbers (top-right = max °C, bottom-right = min °C, which vary per image), plus corner text (`°C` label, lock icons). Some images also have cage-bar grid lines overlapping the cat.
- **Digital** images (527) — plain RGB reference photos, not used in the paper's technique and not used here.
- Split across `Controlled_environment_(Indoor)` (majority, ~2,085 images / 78 cats) and `Uncontrolled_environment_(Outdoor)/{Day,Night}` (~341 images / ~16 cats).
- Each cat is labeled `Healthy` or `Sick` via its containing folder.

**Training data scope:** all thermal images, indoor and outdoor, day and night (per decision — accepts more noise from outdoor lighting variation in exchange for more data and better generalization).

## Architecture

```
service/
  pawduty_ml/
    preprocessing.py   # mask colorbar/text/cage-grid -> isolate cat body -> temp-stat features
    dataset.py         # walk ml-heat/, yield (image_path, cat_id, label)
    train.py           # CLI: fit + evaluate + save model artifact
    model.py           # load artifact, predict(features) -> (tag, confidence)
  api/
    main.py            # FastAPI app, POST /scan
    schema.py          # request/response pydantic models
  models/               # trained artifact (joblib), checked in
  tests/
    fixtures/           # a few real sample images for tests
    test_preprocessing.py
    test_api.py
```

`preprocessing.py` is shared by both the training pipeline and the API so the exact same image transform is used at train and serve time — using two different implementations would risk train/serve skew that silently degrades accuracy.

## Preprocessing & feature extraction

For each thermal JPG:

1. **Mask irrelevant regions** — crop/blank the color-bar column and the corner text overlays (fixed regions, consistent camera-app layout).
2. **Isolate the cat's body** — grayscale, binary threshold against the black background to produce a body mask. If no plausible body region is found (e.g. threshold yields an empty or near-full mask), this image is rejected (see Error Handling).
3. **Recover real temperature values** — OCR the two printed calibration numbers (pytesseract) to get this image's actual min/max °C, then map each pixel's position on the colormap to an interpolated Celsius value between them.
   - **Fallback:** if OCR can't confidently parse both numbers, fall back to normalized 0–1 relative pixel intensity for that image only (logged as a warning, not fatal). This is a deliberate addition beyond the paper's method, since we only have colorized overlay JPGs, not raw radiometric data — losing calibration on a few noisy images shouldn't crash the pipeline.
4. **Compute the feature vector** — from the histogram of temperature values within the body mask: **average, min, max, and most-frequent (mode)** temperature. This matches the paper's 4-feature vector exactly.

## Model & training

- **Model:** `StandardScaler` + `LogisticRegression` (scikit-learn), matching the paper's baseline.
- **Split:** 80/20, split **by Cat ID**, not by image. Splitting by image would let the same cat's images appear in both train and test, leaking identity rather than testing generalization to unseen cats — this is a deliberate improvement over ambiguity in the paper's stated methodology.
- **Evaluation:** accuracy, precision, recall, F1, specificity — same metrics the paper reports, to allow direct comparison (paper baseline: 73.42% accuracy).
- **Training entrypoint:** `python -m pawduty_ml.train`, walks the dataset, extracts features for every thermal image, fits the model, prints the evaluation report, and saves the artifact to `service/models/cat_thermal_lr.joblib`.

## Serving (the worker)

`POST /scan` (multipart image upload):

- Runs the same `preprocessing.py` path as training to get the 4-feature vector.
- Loads `cat_thermal_lr.joblib`, calls `scaler.transform` → `model.predict_proba`.
- Returns `{"tag": "healthy" | "unhealthy", "confidence": <float 0-1>}`.
- If the body-mask step can't detect a cat at all, returns HTTP 422 with a clear error message rather than guessing a tag.

This response shape is intentionally simple JSON so a future client (mobile/web) can call it directly without changes.

## Error handling

- **No cat body detected** in the thermal image → 422, explicit error, no guess.
- **OCR calibration failure** on a given image → fall back to normalized relative-intensity features for that image, log a warning, continue (applies at both training and serving time).
- **Non-image / corrupt upload** → 400 from FastAPI's validation.

## Testing

- **Unit tests** for `preprocessing.py` using a handful of real fixture images (one indoor healthy, one indoor sick, one outdoor) — assert the body mask is plausible (non-empty, bounded) and the feature vector is in a sane range.
- **Unit test** for the OCR-fallback branch (simulate OCR raising or returning unparseable text).
- **API integration test** using FastAPI's `TestClient`, posting a known sample image to `/scan` and asserting the response schema and a plausible tag.
- **Full training run** is manual via the CLI (dataset is small — should train in well under a minute) rather than an automated test; the automated tests cover pipeline correctness, not model accuracy.

## Dependencies

`opencv-python`, `numpy`, `scikit-learn`, `joblib`, `pytesseract` (+ system `tesseract-ocr`), `Pillow`, `fastapi`, `uvicorn`, `pytest`, `httpx`.
