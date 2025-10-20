# ===== runtime image: no training, just serve =====
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/

EXPOSE 8000

# HEALTHCHECK --interval=10s --timeout=3s --start-period=5s CMD \
#   wget -qO- http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn","src.serve:app","--host","0.0.0.0","--port","8000"]
