"""
utils.py
========
Shared utilities: seeding, logging setup, label parsing.
"""

import logging
import os
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Set seeds for Python random, NumPy and PyTorch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic cudnn (slightly slower; ensures reproducibility)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes to stdout with a standard format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def parse_class_name(folder_name: str) -> tuple[str, str]:
    """
    Parse a PlantVillage folder name into (crop, condition).

    Folder format: "<Crop>___<Condition>"
    Split on the FIRST "___" only.

    Returns:
        (crop, condition) both as stripped lowercase strings.

    Example:
        "Tomato___Early_blight" → ("tomato", "early_blight")
        "Cherry_(including_sour)___Powdery_mildew" → ("cherry_(including_sour)", "powdery_mildew")
    """
    parts = folder_name.split("___", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse class name: {folder_name!r}")
    crop = parts[0].strip().lower()
    condition = parts[1].strip().lower()
    return crop, condition


def ensure_dir(path: str | Path) -> Path:
    """Create directory and all parents; return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
