"""Generate the model card and datasheet from committed, sanitized snapshots only."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "reports" / "snapshot"
MODEL_CARD = ROOT / "docs" / "MODEL_CARD.md"
DATASHEET = ROOT / "docs" / "DATASHEET.md"
SENSITIVE_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|C:\\\\Users\\\\Vedant)", re.IGNORECASE)
SENSITIVE_USERNAME = re.compile(r"Vedant", re.IGNORECASE)
LONG_HEX_IDENTIFIER = re.compile(r"(?<![0-9a-f])[0-9a-f]{24,}(?![0-9a-f])", re.IGNORECASE)


def _read_snapshot(path: Path) -> Any | None:
    if not path.is_file():
        return None
    snapshot_text = path.read_text(encoding="utf-8", errors="replace")
    if (
        SENSITIVE_PATH.search(snapshot_text)
        or SENSITIVE_USERNAME.search(snapshot_text)
        or LONG_HEX_IDENTIFIER.search(snapshot_text)
    ):
        raise ValueError(f"Snapshot contains an absolute Windows path or username: {path.name}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return path.read_text(encoding="utf-8")


def _source_date(payload: Any) -> str:
    keys = {"generated_at", "created_at", "retrieved_on", "source_date", "date", "timestamp"}
    found: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key.lower() in keys and isinstance(child, (str, int, float)) and child:
                    found.append(str(child))
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return found[0] if found else "date not recorded in snapshot"


def _compact(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _corn_weakness(payload: Any) -> list[Any]:
    matches: list[Any] = []
    seen: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
            lowered = encoded.lower().replace("_", " ")
            if "corn" in lowered and ("cercospora" in lowered or "gray leaf spot" in lowered):
                if encoded not in seen:
                    matches.append(value)
                    seen.add(encoded)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return matches


def _section(filename: str, title: str, payload: Any | None) -> str:
    if payload is None:
        return f"## {title}\n\nNot generated: missing `reports/snapshot/{filename}`.\n"
    data_date = _source_date(payload)
    if isinstance(payload, dict):
        items = [f"- `{key}`: `{_compact(value)}`" for key, value in payload.items()]
        detail = "\n".join(items) or "- Snapshot contains no fields."
    else:
        detail = f"```text\n{_compact(payload)}\n```"
    return (
        f"## {title}\n\nSource file: `reports/snapshot/{filename}`\n"
        f"Source date: {data_date}\n\n{detail}\n"
    )


def generate(snapshot_dir: Path = SNAPSHOT_DIR) -> tuple[str, str]:
    """Return model-card and datasheet text without reading runtime reports or weights."""
    names = [
        "eval_summary.json",
        "robustness_three_fill.json",
        "robustness_metrics.json",
        "seed_variance.json",
        "bg_check_rates.json",
        "drift_latest.json",
        "promotion_log.jsonl",
        "registry_export.json",
        "advisory_audit.json",
    ]
    data = {name: _read_snapshot(snapshot_dir / name) for name in names}
    candidates = sorted(snapshot_dir.glob("candidate_gate_*.json"))
    candidate_payloads = [(p.name, _read_snapshot(p)) for p in candidates]

    parts = [
        "# Model Card — Plant Disease Pipeline\n",
        "\nGenerated from committed sanitized files in `reports/snapshot/` only. "
        "This engineering prototype is not field validated and is not a diagnostic authority.\n",
        "\n## Model and intended use\n\n"
        "The repository config identifies a ResNet-18 image classifier. It is intended for "
        "course demonstration and engineering evaluation. It is not validated for field use, "
        "commercial decisions, or safety-critical diagnosis. Source: `configs/config.yaml`, "
        "`src/model.py`. Source date: not recorded in those files.\n",
    ]
    for filename, title in (
        ("eval_summary.json", "Evaluation summary"),
        ("robustness_three_fill.json", "Three-fill robustness"),
        ("seed_variance.json", "Seed variance (descriptive only)"),
        ("bg_check_rates.json", "Background sensitivity"),
        ("drift_latest.json", "Latest drift snapshot"),
        ("promotion_log.jsonl", "Promotion history"),
        ("registry_export.json", "Registry snapshot"),
        ("advisory_audit.json", "Advisory provenance coverage"),
    ):
        parts.append(_section(filename, title, data[filename]))
    if data["eval_summary.json"] is None:
        parts.append(_section("eval_summary.json", "Corn weak-class evidence", None))
    else:
        corn_records = _corn_weakness(data["eval_summary.json"])
        parts.append(_section("eval_summary.json", "Corn weak-class evidence", {
            "records": corn_records,
        }) if corn_records else (
            "## Corn weak-class evidence\n\nSource file: "
            "`reports/snapshot/eval_summary.json`  \nSource date: "
            f"{_source_date(data['eval_summary.json'])}\n\n"
            "No matching Corn Cercospora / Gray leaf spot record was found in this snapshot.\n"
        ))
    if candidate_payloads:
        parts.append("## Candidate gate reports\n")
        for filename, payload in candidate_payloads:
            parts.append(_section(filename, "Candidate gate: " + filename, payload))
    else:
        parts.append(
            "## Candidate gate reports\n\nNot generated: no `candidate_gate_*.json` "
            "snapshot exists.\n"
        )
    parts.append(
        "## Interpretation and limitations\n\n"
        "- Validation and test results describe the saved evaluation protocol only; they do not "
        "establish field performance. The available image data are lab-style images, and field "
        "validation has not been established.\n"
        "- The black-fill background ablation is sensitive to the segmentation mask and fill "
        "artifact; it is not proof of shortcut learning. Compare all three fills, including "
        "per-image mean.\n"
        "- A class-level weakness is described only when its record is present in the saved "
        "evaluation snapshot.\n"
        "- Serving reports unscaled softmax confidence (T=1). Temperature fitting is a "
        "diagnostic step; the serving path does not apply fitted temperature scaling. See "
        "`src/api.py` and `docs/DECISIONS.md`.\n"
        "- PlantVillage dataset license and redistribution rights must be verified before any "
        "public release. Sources: `docs/ADVISORY_AUDIT.md`, `src/api.py`, and the cited snapshot "
        "sections above; source dates are not recorded unless shown with a snapshot section.\n"
    )
    card = "\n".join(parts).rstrip() + "\n"

    datasheet_parts = [
        "# Dataset Datasheet\n",
        "\nThis datasheet is generated only from `reports/snapshot/` and tracked project "
        "configuration. Missing evidence is stated as missing; no dataset history or license "
        "terms are inferred.\n",
        _section("eval_summary.json", "Saved evaluation data", data["eval_summary.json"]),
        _section("seed_variance.json", "Seed summary", data["seed_variance.json"]),
        "## Data characteristics and limitations\n\n"
        "The project evaluates plant-leaf image classes using a PlantVillage-derived dataset "
        "as named in the project configuration and documentation. The available data are "
        "lab-style images; field representativeness is not established. A missing evaluation "
        "snapshot means counts and class distributions are not generated in this card.\n\n"
        "## Collection, consent, and license\n\n"
        "Collection methods, consent details, and redistribution terms are not established by "
        "the committed snapshot inputs. Verify the dataset license and permitted uses before "
        "public release.\n",
    ]
    datasheet = "\n".join(datasheet_parts).rstrip() + "\n"
    return card, datasheet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if generated docs differ")
    parser.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT_DIR)
    parser.add_argument("--model-card", type=Path, default=MODEL_CARD)
    parser.add_argument("--datasheet", type=Path, default=DATASHEET)
    args = parser.parse_args()
    card, datasheet = generate(args.snapshot_dir)
    expected = ((args.model_card, card), (args.datasheet, datasheet))
    if args.check:
        stale = [
            str(path)
            for path, text in expected
            if not path.is_file() or path.read_text(encoding="utf-8") != text
        ]
        if stale:
            print("Generated documentation is stale: " + ", ".join(stale))
            return 1
        print("Model card and datasheet are up to date.")
        return 0
    for path, text in expected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"Wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
