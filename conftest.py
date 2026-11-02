"""
conftest.py
===========
Shared pytest fixtures for the plant_disease test suite.
Sets DATA_ROOT from .env before any test module is imported.
"""

import sys
import os
from pathlib import Path

# Ensure src/ is importable from tests/
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Load .env early so DATA_ROOT is set before config_loader is imported
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed; expect DATA_ROOT in env

def api_artifact_skip_reason(project_root: Path | None = None) -> str | None:
    """Return a skip reason when local API artifacts/configuration are unavailable."""
    root = project_root or Path(__file__).resolve().parent
    checkpoint = root / "models" / "best_model.pth"
    if not checkpoint.exists():
        return "Model artifact unavailable; API integration tests require local weights"

    data_root = os.environ.get("DATA_ROOT", "").strip()
    if not data_root or not Path(data_root).exists():
        return "DATA_ROOT unavailable; API integration tests require the local dataset"

    return None


if not os.environ.get("DATA_ROOT", "").strip():
    os.environ["DATA_ROOT"] = str(project_root / "data")
