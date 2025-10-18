from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import joblib, json
from pathlib import Path

# ---- 输入数据模式（10 个特征）----
class DiabetesFeatures(BaseModel):
    age: float = Field(...)
    sex: float = Field(...)
    bmi: float = Field(...)
    bp: float = Field(...)
    s1: float = Field(...)
    s2: float = Field(...)
    s3: float = Field(...)
    s4: float = Field(...)
    s5: float = Field(...)
    s6: float = Field(...)

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
    # 按要求返回 JSON
    return {"status": "ok", "model_version": _model_version}

@app.post("/predict")
def predict(features: DiabetesFeatures):
    model = get_model()
    x = [[
        features.age, features.sex, features.bmi, features.bp,
        features.s1, features.s2, features.s3, features.s4,
        features.s5, features.s6
    ]]
    y_hat = float(model.predict(x)[0])
    return {"prediction": y_hat}
