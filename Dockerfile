# Dockerfile — Plant Disease Pipeline
# Multi-stage, multi-target build.
# Targets:
#   api  — FastAPI + ResNet18 inference (CPU, no dataset required)
#   ui   — Streamlit dashboard
#
# Build examples:
#   docker build --target api -t plant-api .
#   docker build --target ui  -t plant-ui  .
#   docker compose up          (builds both via docker-compose.yml)

# ── Shared base ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install shared Python deps (CPU PyTorch for containerised serving)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    fastapi==0.143.0 \
    uvicorn[standard]==0.54.0 \
    python-multipart==0.0.32 \
    pillow==12.3.0 \
    pyyaml==6.0.3 \
    numpy==2.5.3 \
    scipy==1.18.1 \
    matplotlib==3.11.2 \
    seaborn==0.13.2 \
    scikit-learn==1.7.1 \
    mlflow==3.17.1 \
    python-dotenv==1.2.1

COPY src/ /app/src/
COPY configs/ /app/configs/
COPY kb/ /app/kb/

# ── API target ────────────────────────────────────────────────────────────────
FROM base AS api

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ── UI target ─────────────────────────────────────────────────────────────────
FROM base AS ui

RUN pip install --no-cache-dir \
    streamlit==1.45.1 \
    pandas==2.3.1 \
    requests==2.32.4

COPY app/ /app/app/
COPY demo_samples/ /app/demo_samples/
COPY reports/ /app/reports/

EXPOSE 8501

HEALTHCHECK --interval=20s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/ui.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
