# Makefile for Plant Disease Pipeline
# Compatible with: Windows (GNU Make 3.82+) and Unix make.
# Usage (from the plant_disease/ directory):
#   make up          — start API + UI via docker compose (recommended for demo)
#   make serve       — start API directly (no Docker)
#   make ui          — start Streamlit UI directly (no Docker)
#   make evaluate    — run full evaluation suite, regenerate reports/metrics.json
#   make test        — run pytest (18 tests)
#   make drift       — run drift simulation demo
#   make promote     — run promotion gate

PYTHON     = ..\\.venv\\Scripts\\python.exe
PIP        = ..\\.venv\\Scripts\\pip.exe
PYTEST     = ..\\.venv\\Scripts\\pytest.exe
UVICORN    = ..\\.venv\\Scripts\\uvicorn.exe
STREAMLIT  = ..\\.venv\\Scripts\\streamlit.exe

.PHONY: all setup verify-gpu audit train serve ui evaluate promote test drift up down clean

# ── Default ────────────────────────────────────────────────────────────────────
all: setup verify-gpu

# ── Install dependencies ───────────────────────────────────────────────────────
setup:
	$(PIP) install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
	$(PIP) install -r requirements.txt

# ── Verify GPU is available ────────────────────────────────────────────────────
verify-gpu:
	$(PYTHON) -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('GPU OK:', torch.cuda.get_device_name(0))"

# ── Step 2: Audit, dedupe, split ──────────────────────────────────────────────
audit:
	$(PYTHON) scripts/audit_and_split.py

# ── Step 3: Train (full data, 8 epochs) ───────────────────────────────────────
train:
	$(PYTHON) scripts/train.py

# ── Step 4a: Start FastAPI backend directly (development) ─────────────────────
serve:
	$(UVICORN) src.api:app --host 127.0.0.1 --port 8000

# ── Step 4b: Launch Streamlit UI directly (development) ───────────────────────
ui:
	$(STREAMLIT) run app/ui.py --server.port 8501

# ── One-command stack via Docker Compose (recommended for demo) ───────────────
up:
	docker compose up --build -d
	@echo "API:  http://localhost:8000/health"
	@echo "UI:   http://localhost:8501"
	@echo "Use 'make down' to stop."

down:
	docker compose down

# ── Full evaluation suite (regenerates reports/metrics.json + figures) ─────────
evaluate:
	$(PYTHON) scripts/evaluate.py

# ── Promotion gate ─────────────────────────────────────────────────────────────
promote:
	$(PYTHON) scripts/promote.py

# ── Run all tests ──────────────────────────────────────────────────────────────
test:
	$(PYTEST) tests/ -v

# ── Drift simulation demo ──────────────────────────────────────────────────────
drift:
	$(PYTHON) scripts/simulate_drift.py

# ── Clean generated artifacts (not models or data) ────────────────────────────
clean:
	if exist mlflow.db del mlflow.db
	if exist logs rmdir /s /q logs
	if exist reports\*.png del /q reports\*.png
	if exist reports\*.html del /q reports\*.html
