from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path


class DiabetesFeatures(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


app = FastAPI(title="Diabetes Triage Service", version="0.1")

MODEL_PATH = Path("models/model.pkl")
VERSION_PATH = Path("models/model_version.txt")

_model = None
_model_version = "unknown"


def get_model():
    global _model, _model_version
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        if VERSION_PATH.exists():
            _model_version = VERSION_PATH.read_text().strip()
    return _model


@app.get("/health")
def health():
    global _model_version
    if _model is None and VERSION_PATH.exists():
        _model_version = VERSION_PATH.read_text().strip()
    return {"status": "ok", "model_version": _model_version}


@app.post("/predict")
def predict(features: DiabetesFeatures):
    model = get_model()
    x = [
        [
            features.age,
            features.sex,
            features.bmi,
            features.bp,
            features.s1,
            features.s2,
            features.s3,
            features.s4,
            features.s5,
            features.s6,
        ]
    ]
    y_hat = float(model.predict(x)[0])
    return {"prediction": y_hat}
