# diabetes-triage-mlops
mlop assignment 3

# Diabetes Triage MLOps (v0.1)

## Local dev
```bash
python -m venv .venv && source .venv/bin/activate   # Windows 用 .venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
uvicorn src.serve:app --host 0.0.0.0 --port 8000
# open http://127.0.0.1:8000/health
