import numpy as np

from pawduty_ml.train import evaluate_model


def test_evaluate_model_reports_expected_metric_keys() -> None:
    # 4 groups (cats), perfectly separable by the first feature. GroupKFold(2)
    # round-robins the groups so each training fold sees both classes, and the
    # separable signal is recovered in every fold.
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

    report = evaluate_model(X, y, groups, n_splits=2)

    assert set(report) == {"accuracy", "precision", "recall", "f1", "specificity"}
    for value in report.values():
        assert 0.0 <= value <= 1.0
    # Cross-validated accuracy on perfectly separable data should be perfect.
    assert report["accuracy"] == 1.0
