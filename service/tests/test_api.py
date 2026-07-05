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
