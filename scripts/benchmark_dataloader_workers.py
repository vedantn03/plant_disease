"""Small Windows DataLoader worker probe; does not train or modify artifacts."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config
from src.dataset import PlantVillageDataset, get_transforms
from src.utils import set_seed


def measure_workers(workers: int, batches: int, cfg, split_path: Path) -> dict:
    set_seed(cfg.seed)
    dataset = PlantVillageDataset(
        split_path,
        transform=get_transforms("val", cfg),
        seed=cfg.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    labels = []
    start = time.perf_counter()
    for index, (_, batch_labels) in enumerate(loader):
        labels.extend(batch_labels.tolist())
        if index + 1 >= batches:
            break
    elapsed = time.perf_counter() - start
    return {
        "workers": workers,
        "seconds": round(elapsed, 3),
        "batches": min(batches, len(loader)),
        "images": len(labels),
        "images_per_second": round(len(labels) / max(elapsed, 1e-9), 3),
        "first_labels": labels[:16],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--batches", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config()
    split_path = Path(cfg.split.manifest_dir) / "val.csv"
    first = [measure_workers(w, args.batches, cfg, split_path) for w in args.workers]
    second = [measure_workers(w, args.batches, cfg, split_path) for w in args.workers]
    for a, b in zip(first, second):
        print({
            "workers": a["workers"],
            "first_seconds": a["seconds"],
            "second_seconds": b["seconds"],
            "deterministic_labels": a["first_labels"] == b["first_labels"],
        })


if __name__ == "__main__":
    main()
