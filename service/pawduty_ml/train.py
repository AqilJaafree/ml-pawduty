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
