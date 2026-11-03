"""Send unlabelled field images to the API and summarize returned outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config

IMAGE_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def summarize_responses(responses: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(item.get("status", "unknown")) for item in responses)
    abstain_reasons = Counter(
        str(item.get("reason_code") or item.get("message") or "unspecified")
        for item in responses
        if item.get("status") in {"uncertain", "rejected_not_a_leaf"}
    )
    bg_checks = Counter(str(item.get("bg_check", "unknown")) for item in responses)
    predicted_classes: Counter[str] = Counter()
    for item in responses:
        crop = item.get("crop")
        condition = item.get("condition")
        predicted_class = item.get("predicted_class")
        if predicted_class:
            predicted_classes[str(predicted_class)] += 1
        elif crop and condition:
            predicted_classes[f"{crop}___{condition}"] += 1

    return {
        "title": "indicative only; no ground truth",
        "label_status": "UNLABELLED",
        "image_count": len(responses),
        "status_counts": dict(sorted(statuses.items())),
        "abstain_reason_counts": dict(sorted(abstain_reasons.items())),
        "bg_check_counts": {
            result: bg_checks[result]
            for result in ("consistent", "inconsistent", "unknown")
        },
        "top_predicted_classes": [
            {"class": name, "count": count}
            for name, count in predicted_classes.most_common()
        ],
        "traffic_note": "Field photos are unlabelled observations; this report has no ground truth.",
    }


def run_field_probe(
    image_dir: Path,
    *,
    api_url: str,
    timeout_seconds: float,
    report_path: Path,
    post: Callable[..., Any] = requests.post,
) -> dict[str, Any]:
    """Submit every supported image file in a directory and write a JSON summary."""
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_TYPES
    )
    if not image_paths:
        raise ValueError(f"No supported images found in {image_dir}")

    responses: list[dict[str, Any]] = []
    for image_path in image_paths:
        try:
            with image_path.open("rb") as image_file:
                response = post(
                    f"{api_url.rstrip('/')}/predict",
                    files={
                        "file": (
                            image_path.name,
                            image_file,
                            IMAGE_TYPES[image_path.suffix.lower()],
                        )
                    },
                    timeout=timeout_seconds,
                )
            response.raise_for_status()
            result = response.json()
            responses.append(result)
        except Exception:
            responses.append({
                "status": "error",
                "bg_check": "unknown",
            })

    report = summarize_responses(responses)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    cfg = load_config(PROJECT_ROOT / "configs" / "config.yaml", require_data_root=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default=str(cfg.background_check.field_probe_dir))
    parser.add_argument(
        "--api-url", default=f"http://127.0.0.1:{int(cfg.serving.port)}"
    )
    parser.add_argument(
        "--output", default=str(cfg.background_check.field_probe_report_path)
    )
    args = parser.parse_args()
    try:
        report = run_field_probe(
            PROJECT_ROOT / args.directory,
            api_url=args.api_url,
            timeout_seconds=float(cfg.background_check.request_timeout_seconds),
            report_path=PROJECT_ROOT / args.output,
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps({key: value for key, value in report.items() if key != "responses"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
