"""Batch-run the cat thermal health model over a folder of images.

Usage:
    python scan_samples.py [folder] [--temp-range MIN MAX]

The ml-heat model expects °C-scale features. Images without on-frame calibration
badges (e.g. phone/WhatsApp thermal photos) OCR to nothing, so the pipeline falls
back to raw 0-1 colormap positions and the model sees garbage. Passing --temp-range
supplies the missing min/max °C so the 0-1 positions are affinely mapped to real
temperatures before classification, matching the training distribution.
"""
import argparse
from pathlib import Path

import pawduty_ml.preprocessing as pp
from pawduty_ml.model import load_model, predict
from pawduty_ml.preprocessing import (
    NoBodyDetectedError,
    extract_temperature_features,
    features_to_vector,
)

parser = argparse.ArgumentParser()
parser.add_argument("folder", nargs="?", default="../sample-image")
parser.add_argument(
    "--temp-range",
    nargs=2,
    type=float,
    metavar=("MIN", "MAX"),
    help="manual °C calibration range applied when the image has no OCR badges",
)
args = parser.parse_args()

folder = Path(args.folder)
images = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

# Inject a manual calibration range: monkeypatch read_calibration_range so the
# existing (tested) feature pipeline treats these images as calibrated. This only
# affects the affine value-scaling step; body isolation is unchanged.
if args.temp_range:
    lo, hi = args.temp_range
    pp.read_calibration_range = lambda image: (lo, hi)
    print(f"Manual calibration: {lo}–{hi} °C\n")
else:
    print("No manual calibration (raw 0-1 colormap positions)\n")

scaler, model = load_model()
print(f"Scanning {len(images)} images in {folder}\n")
print(f"{'image':<52} {'tag':<10} {'conf':>6}  {'calib':>5}   features [avg,min,max,mode]")
print("-" * 112)

tally = {"healthy": 0, "unhealthy": 0}
for img in images:
    try:
        feats = extract_temperature_features(img)
        vec = features_to_vector(feats)
        result = predict(vec, scaler, model)
        tally[result.tag] += 1
        fv = ", ".join(f"{x:.1f}" for x in vec)
        flag = "yes" if feats.calibrated else "no"
        print(f"{img.name:<52} {result.tag:<10} {result.confidence:>6.2f}  {flag:>5}   [{fv}]")
    except NoBodyDetectedError as exc:
        print(f"{img.name:<52} {'NO BODY':<10} {'—':>6}  {'—':>5}   ({exc})")
    except Exception as exc:  # noqa: BLE001
        print(f"{img.name:<52} {'ERROR':<10} {'—':>6}  {'—':>5}   ({type(exc).__name__}: {exc})")

print("-" * 112)
print(f"Totals: {tally['healthy']} healthy, {tally['unhealthy']} unhealthy")
