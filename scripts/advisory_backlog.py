"""Rank frequently requested classes that lack VERIFIED advisory evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config  # noqa: E402


def build_backlog(log_path: Path, kb_path: Path) -> list[dict]:
    kb = yaml.safe_load(kb_path.read_text(encoding="utf-8")) or {}
    grade_ok = {
        f"{entry['crop']}___{entry['condition']}"
        for entry in kb.get("advisories", [])
        if entry.get("status") == "VERIFIED" and entry.get("source_url")
    }
    counts: Counter[str] = Counter()
    replay: Counter[str] = Counter()
    if log_path.exists():
        with log_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                label = row.get("predicted_class")
                if label and label not in grade_ok:
                    counts[label] += 1
                    if row.get("traffic_source") == "replay":
                        replay[label] += 1
    return [
        {"class": label, "requests": count, "replay_requests": replay[label]}
        for label, count in counts.most_common()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--kb", type=Path, default=ROOT / "kb" / "advisories.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = load_config(require_data_root=False)
    logs = args.logs or ROOT / cfg.advisory.request_log_path
    output = args.output or ROOT / cfg.advisory.backlog_report_path
    logs = logs if logs.is_absolute() else ROOT / logs
    output = output if output.is_absolute() else ROOT / output
    kb = args.kb if args.kb.is_absolute() else ROOT / args.kb
    result = build_backlog(logs, kb)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "classes": len(result)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
