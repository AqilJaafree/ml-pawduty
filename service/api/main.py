import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.schema import ScanResponse
from pawduty_ml.model import load_model, predict
from pawduty_ml.preprocessing import NoBodyDetectedError, extract_temperature_features, features_to_vector

app = FastAPI(title="Pawduty Cat Thermal Health Worker")

# Open CORS so the browser-based web client can call /scan cross-origin (dev/demo).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        except Exception as exc:  # noqa: BLE001 - any decode/processing failure
            # Return a clean 4xx (which keeps CORS headers, unlike an unhandled
            # 500) so the browser can read the message instead of reporting the
            # service as unreachable. The model needs a thermal (INFERNO-colormap)
            # frame with °C badges — a normal photo lands here.
            raise HTTPException(
                status_code=422,
                detail="Could not read a thermal image from that photo. Use a thermal (INFERNO-colormapped) frame with the temperature badges.",
            ) from exc

    vector = features_to_vector(features)
    result = predict(vector, scaler, model)
    return ScanResponse(tag=result.tag, confidence=result.confidence)
