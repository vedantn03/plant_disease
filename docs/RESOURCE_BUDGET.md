# Measured System Resource Budget (T2)

Generated from genuine hardware inspection via Windows system calls, `nvidia-smi`, and socket probes.

---

## 1. Host Hardware Specifications

| Resource | Value / Device | Notes |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 (AMD64) | Tested on local development host |
| **CPU Architecture** | x86_64 | Multi-core host system |
| **Total Physical RAM** | **23.71 GB** | Measured via `GlobalMemoryStatusEx` (`25,458,548,736` bytes) |
| **Available RAM (Idle)** | **13.00 GB** | Adequate buffer for local serving, monitoring & Airflow containers |
| **GPU Model** | **NVIDIA GeForce RTX 3050 Laptop GPU** | Volatile VRAM: **6,144 MiB (6.0 GB)** |
| **NVIDIA Driver Version** | **581.86** | CUDA Version: 13.0 (compatible with PyTorch `cu128`) |
| **Python Version** | **3.14.7** | Virtual environment at `..\.venv` |

---

## 2. Docker & Containerization Resources

| Component | Status | Limit / Configuration |
| :--- | :--- | :--- |
| **Docker CLI** | `v29.8.0` (Client) | Installed under `%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin\docker.EXE` |
| **Docker Compose** | `v5.5.1` (Plugin) | Docker Compose plugin operational |
| **Docker Daemon Engine** | Offline / Suspended | Must start Docker Desktop before running container profiles |
| **WSL2 Memory Target** | ~8.0 GB - 12.0 GB recommended | Configured via `.wslconfig` on host to prevent memory starvation |

---

## 3. Port Allocation Table

Probed on loopback interface `127.0.0.1`:

| Port | Service Assigned | Status at Probe | Conflict Resolution |
| :--- | :--- | :--- | :--- |
| **5000** | MLflow UI | **FREE** | None required |
| **8000** | FastAPI Prediction Backend | **FREE** | Standard serving port |
| **8501** | Streamlit Web Interface | **FREE** | Front-end UI |
| **9090** | Prometheus Metrics Scraper | **FREE** | Observability scraper |
| **3000** | Grafana Telemetry Dashboard | **FREE** | Visualization panels |
| **8080** | Airflow Webserver | **FREE** | Orchestration DAG interface |
| **9000** | MinIO Object Store API (S3) | **FREE** | S3 artifact store |

---

## 4. Container Profile Memory Allocation Matrix

| Compose Profile | Services Included | Projected RAM Consumption | Feasibility on 23.7 GB Host |
| :--- | :--- | :--- | :--- |
| **`core`** | `mlflow`, `api` (FastAPI), `ui` (Streamlit) | ~1.5 GB | **High (Pass)** |
| **`monitoring`** | `prometheus`, `grafana` | ~600 MB | **High (Pass)** |
| **`storage`** | `minio` (S3 object store) | ~400 MB | **High (Pass)** |
| **`orchestration`**| `airflow-webserver`, `airflow-scheduler`, `postgres` | ~2.5 GB - 3.0 GB | **High (Pass)** |
| **Full Stack** | All 8 services active simultaneously | ~5.5 GB - 6.0 GB | **Well within 13.0 GB free RAM** |
