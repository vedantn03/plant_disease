"""
config_loader.py
================
Loads configs/config.yaml and merges with environment variables.

Usage:
    from src.config_loader import load_config
    cfg = load_config()
    data_root = cfg.data.root
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


class _AttrDict(dict):
    """Dict with attribute-style access, recursively applied."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            return _AttrDict(val) if isinstance(val, dict) else val
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(
    config_path: str | Path | None = None,
    *,
    require_data_root: bool = True,
) -> _AttrDict:
    """
    Load YAML config and apply env-var overrides.

    Priority (highest first):
        1. Environment variables (DATA_ROOT, MLFLOW_TRACKING_URI)
        2. config.yaml values

    Args:
        config_path: Path to the YAML file. Defaults to
                     <project_root>/configs/config.yaml.
        require_data_root: Keep the dataset-path validation enabled for training
                           and audit tools. Serving can disable it because inference
                           reads the mounted checkpoint, not the training dataset.

    Returns:
        _AttrDict with dot-access to nested keys.
    """
    # Load .env if present (won't overwrite existing env vars)
    load_dotenv(dotenv_path=_project_root() / ".env", override=False)

    if config_path is None:
        config_path = _project_root() / "configs" / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh)

    cfg = _AttrDict(raw)

    # ── Env-var overrides ──────────────────────────────────────────────
    data_root = os.environ.get("DATA_ROOT", "").strip()
    if data_root:
        cfg["data"]["root"] = data_root

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if mlflow_uri:
        cfg["mlflow"]["tracking_uri"] = mlflow_uri

    model_source = os.environ.get("SERVING_MODEL_SOURCE", "").strip()
    if model_source:
        cfg["serving"]["model_source"] = model_source

    # Validate that data root is set
    if require_data_root and not cfg["data"]["root"]:
        raise EnvironmentError(
            "DATA_ROOT is not set. Copy .env.example to .env and fill in DATA_ROOT."
        )

    return cfg


def _project_root() -> Path:
    """Return the plant_disease/ project root (parent of src/)."""
    return Path(__file__).resolve().parent.parent
