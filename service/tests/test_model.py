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
