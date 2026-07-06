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
