# Diabetes Triage ML Service

A minimal MLOps project that trains and serves a model predicting short-term diabetes progression risk.  
Built with scikit-learn, FastAPI, Docker, and GitHub Actions.

---

## Setup

```bash
git clone https://github.com/yuzhunn/diabetes-triage-mlops.git
cd diabetes-triage-mlops

python -m venv .venv
. .venv/Scripts/Activate.ps1       # Windows
# source .venv/bin/activate        # macOS/Linux

python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Training

Baseline (v0.1):
```bash
python src/train.py --model linear
```

Improved model (v0.2):
```bash
python src/train.py --model ridge --alpha 30
```

Artifacts are saved to `models/`:
```
models/
  ├── model.pkl
  ├── metrics.json
  └── model_version.txt
```

---

## Run API Service

```bash
python src/serve.py
```

### Health check
```bash
curl.exe http://127.0.0.1:8000/health
```
Response:
```json
{"status":"ok","model_version":"unknown"}
```

### Prediction
```bash
curl.exe -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"age\":0.02,\"sex\":-0.044,\"bmi\":0.06,\"bp\":-0.03,\"s1\":-0.02,\"s2\":0.03,\"s3\":-0.02,\"s4\":0.02,\"s5\":0.02,\"s6\":-0.001}"
```
Response:
```json
{"prediction": 147.53}
```

---

## Docker

Build locally:
```bash
docker build -t diabetes-triage-mlops:latest .
docker run -p 8000:8000 diabetes-triage-mlops:latest
```

Or pull release images:
```bash
docker pull ghcr.io/yuzhunn/diabetes-triage-mlops:v0.1
docker pull ghcr.io/yuzhunn/diabetes-triage-mlops:v0.2
```

---

v0.1: LinearRegression (RMSE 53.8534)  
v0.2: Ridge(alpha=30) (RMSE 53.5004, ~0.6% improvement)
