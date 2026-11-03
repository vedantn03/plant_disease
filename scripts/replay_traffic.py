"""Replay a deterministic sample of real TEST images to a running API as marked replay traffic."""

from __future__ import annotations

import argparse
import csv
import io
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image, ImageEnhance, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def degrade_image(image: Image.Image, mode: str) -> Image.Image:
    """Apply a named image degradation for controlled replay testing."""
    if mode == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=3))
    if mode == "dark":
        return ImageEnhance.Brightness(image).enhance(0.4)
    if mode == "noise":
        import numpy as np

        rng = np.random.default_rng(42)
        values = np.asarray(image.convert("RGB"), dtype=np.int16)
        noise = rng.normal(0, 32, values.shape).astype(np.int16)
        return Image.fromarray(np.clip(values + noise, 0, 255).astype("uint8"))
    raise ValueError(f"Unsupported degradation: {mode}")


def load_test_paths(manifest: Path, limit: int, seed: int) -> list[Path]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    paths = [
        Path(row["path"]) if Path(row["path"]).is_absolute() else PROJECT_ROOT / row["path"]
        for row in rows
    ]
    rng = random.Random(seed)
    rng.shuffle(paths)
    return paths[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--manifest", default="data/splits/test.csv")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rate", type=float, default=1.0, help="Requests per second")
    parser.add_argument("--degrade", choices=("blur", "dark", "noise"))
    parser.add_argument("--workers", type=int, default=4, help="Parallel image preparation workers")
    args = parser.parse_args()
    if args.limit <= 0 or args.rate <= 0:
        parser.error("--limit and --rate must be positive")
    manifest = PROJECT_ROOT / args.manifest
    paths = load_test_paths(manifest, args.limit, args.seed)
    if not paths:
        parser.error(f"No TEST rows found in {manifest}")

    def prepare(path: Path) -> bytes:
        image = Image.open(path).convert("RGB")
        if args.degrade:
            image = degrade_image(image, args.degrade)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return buffer.getvalue()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        payloads = list(pool.map(prepare, paths))
    for index, (path, payload) in enumerate(zip(paths, payloads)):
        response = requests.post(
            f"{args.api_url.rstrip('/')}/predict",
            files={"file": ("replay.jpg", payload, "image/jpeg")},
            data={"traffic_source": "replay"},
            headers={"X-Traffic-Source": "replay"},
            timeout=120,
        )
        print(f"{index + 1}/{len(paths)} status={response.status_code} body={response.text}")
        if index + 1 < len(paths):
            time.sleep(1.0 / args.rate)
    print(f"Replay complete: {len(paths)} requests; traffic_source=replay; degrade={args.degrade or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
