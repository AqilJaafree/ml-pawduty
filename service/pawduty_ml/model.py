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
