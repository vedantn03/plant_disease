"""Copy available report evidence into a size-limited, path-sanitized snapshot folder."""

from __future__ import annotations

import json
import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_tracked_files import configured_max_bytes

SOURCE = ROOT / "reports"
DESTINATION = SOURCE / "snapshot"
STATIC_REPORTS = (
    "eval_summary.json",
    "robustness_three_fill.json",
    "robustness_metrics.json",
    "seed_variance.json",
    "bg_check_rates.json",
    "drift_latest.json",
    "promotion_log.jsonl",
    "registry_export.json",
)
LOCAL_PATH = re.compile(r"(?:[A-Za-z]:[\\/][^\s\"']*|file:(?:/{1,3})[^\s\"']*)", re.IGNORECASE)
USERNAME = re.compile(r"Vedant", re.IGNORECASE)
LONG_HEX_IDENTIFIER = re.compile(r"(?<![0-9a-f])[0-9a-f]{24,}(?![0-9a-f])", re.IGNORECASE)
IDENTIFIER_KEYS = re.compile(
    r"(?:run_id|run_uuid|git_commit|(?:^|_)hash$|_sha256$|_md5$)", re.IGNORECASE
)


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<redacted-identifier>"
            if IDENTIFIER_KEYS.search(str(key)) and isinstance(child, str)
            else _sanitize(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_sanitize(child) for child in value]
    if isinstance(value, str):
        value = USERNAME.sub("<redacted-user>", LOCAL_PATH.sub("<redacted-local-path>", value))
        return LONG_HEX_IDENTIFIER.sub("<redacted-identifier>", value)
    return value


def _write_snapshot(source: Path, target: Path, max_bytes: int) -> None:
    if source.suffix == ".jsonl":
        lines = [
            json.dumps(_sanitize(json.loads(line)), ensure_ascii=False, sort_keys=True)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        text = "\n".join(lines) + ("\n" if lines else "")
        target.write_text(text, encoding="utf-8", newline="\n")
    elif source.suffix == ".json":
        payload = _sanitize(json.loads(source.read_text(encoding="utf-8")))
        target.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    else:
        shutil.copyfile(source, target)
    if target.stat().st_size > max_bytes:
        target.unlink(missing_ok=True)
        raise ValueError(f"Snapshot exceeds {max_bytes} byte limit: {source.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    args = parser.parse_args()
    source_dir = args.source
    destination_dir = args.destination
    destination_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = configured_max_bytes()
    sources = [source_dir / name for name in STATIC_REPORTS]
    sources.extend(sorted(source_dir.glob("candidate_gate_*.json")))
    copied: list[str] = []
    retained: list[str] = []
    missing: list[str] = []
    for source in sources:
        target = destination_dir / source.name
        if source.parent == destination_dir and source.is_file():
            retained.append(source.name)
            continue
        if not source.is_file():
            if target.is_file():
                retained.append(source.name)
                continue
            missing.append(source.name)
            continue
        _write_snapshot(source, target, max_bytes)
        copied.append(source.name)
    for target in sorted(destination_dir.iterdir()):
        if target.suffix.lower() in {".json", ".jsonl"} and target.is_file():
            _write_snapshot(target, target, max_bytes)
    print(json.dumps({
        "snapshot_dir": destination_dir.as_posix(),
        "copied": copied,
        "retained": retained,
        "missing": missing,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
