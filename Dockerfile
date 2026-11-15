# Dockerfile — Plant Disease Pipeline
# Multi-stage, multi-target build.
# Targets:
#   api  — FastAPI + ResNet18 inference (CPU, no dataset required)
#   ui   — static farmer-facing browser dashboard
#
# Build examples:
#   docker build --target api -t plant-api .
#   docker build --target ui  -t plant-ui  .
#   docker compose up          (builds both via docker-compose.yml)

# ── Shared base ───────────────────────────────────────────────────────────────
FROM python:3.12-slim AS base

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
    prometheus-client \
    python-dotenv==1.2.1

COPY src/ /app/src/
COPY configs/ /app/configs/
COPY kb/ /app/kb/
COPY scripts/drift_exporter.py /app/scripts/drift_exporter.py
COPY dvc.yaml /app/dvc.yaml
COPY docs/MODEL_CARD.md /app/docs/MODEL_CARD.md
COPY .github/workflows/ /app/.github/workflows/

# ── API target ────────────────────────────────────────────────────────────────
FROM base AS api

# Explainability packages are API-only. The static UI image remains lean.
RUN pip install --no-cache-dir \
    shap==0.54.1 \
    lime==0.2.0.1 \
    scikit-image==0.26.0

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ── UI target ─────────────────────────────────────────────────────────────────
# Keep the browser dashboard independent of the API's large ML runtime. The
# browser talks directly to the separately built API service.
FROM python:3.12-slim AS ui

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY web/ /app/web/
COPY demo_samples/*.jpg /app/web/demo_samples/

EXPOSE 8501

HEALTHCHECK --interval=20s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/', timeout=3)" || exit 1

CMD ["python", "/app/web/server.py"]
