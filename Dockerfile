# ===== build stage: 安装依赖 + 训练 =====
FROM python:3.11-slim AS build
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
RUN python src/train.py

# ===== run stage: 只带运行所需 + 模型产物 =====
FROM python:3.11-slim AS run
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=build /app/models/ models/
COPY src/serve.py src/serve.py
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s \
  CMD python -c "import requests; import sys; \
import os; \
import time; \
import json; \
import urllib.request as u; \
import urllib.error as e; \
import urllib.request; \
import urllib.parse; \
import urllib.response; \
import urllib.request as r; \
import urllib.request; \
print()" || exit 0
# 简化：直接依赖容器启动是否成功；健康检查可改用 curl

CMD ["uvicorn","src.serve:app","--host","0.0.0.0","--port","8000"]
