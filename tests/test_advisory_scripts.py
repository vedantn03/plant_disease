import csv
import json
import pytest
import numpy as np
import torch
import yaml

from scripts.advisory_backlog import build_backlog
from scripts.derive_confidence_tiers import derive_table
from scripts.export_confused_pairs import export_pairs
from scripts.export_val_predictions import export_predictions


def test_confidence_table_uses_saved_values_and_input_hash(tmp_path):
    path = tmp_path / "val.csv"
    path.write_text(
        "predicted_class,confidence,correct\n"
        "tomato___healthy,0.4,false\n"
        "tomato___healthy,0.8,true\n"
        "corn___healthy,0.7,true\n",
        encoding="utf-8",
    )
    result = derive_table(path, [0.50, 0.75])
    assert len(result["input_sha256"]) == 64
    assert result["rows"]["tomato"][0]["cutoff"] == pytest.approx(0.6)
    assert result["rows"]["tomato"][0]["precision"] == 1.0


def test_confidence_script_refuses_missing_prediction_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="VAL prediction file not found"):
        derive_table(tmp_path / "missing.csv", [0.50])


def test_backlog_ranks_only_missing_evidence_and_marks_replay(tmp_path):
    kb_path = tmp_path / "advisories.yaml"
    kb_path.write_text(yaml.safe_dump({"advisories": [{
        "crop": "tomato", "condition": "healthy", "status": "VERIFIED",
        "source_url": "https://example.org",
    }]}), encoding="utf-8")
    log_path = tmp_path / "requests.jsonl"
    rows = [
        {"predicted_class": "tomato___healthy", "traffic_source": "api"},
        {"predicted_class": "corn___rust", "traffic_source": "replay"},
        {"predicted_class": "corn___rust", "traffic_source": "api"},
    ]
    log_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert build_backlog(log_path, kb_path) == [
        {"class": "corn___rust", "requests": 2, "replay_requests": 1}
    ]


def test_export_confused_pairs_uses_only_saved_evaluation_fields(tmp_path):
    summary = tmp_path / "eval_summary.json"
    summary.write_text(json.dumps({"top_10_confused_pairs": [
        {"true": "A", "pred": "B", "count": 3}
    ]}), encoding="utf-8")
    output = tmp_path / "confused_pairs.csv"
    assert export_pairs(summary, output) == 1
    with output.open(encoding="utf-8", newline="") as handle:
        assert next(csv.DictReader(handle)) == {
            "true_label": "A", "predicted_label": "B", "count": "3"
        }


def test_export_val_predictions_uses_saved_arrays_and_checkpoint_map(tmp_path):
    logits = tmp_path / "val_logits.npy"
    labels = tmp_path / "val_labels.npy"
    checkpoint = tmp_path / "checkpoint.pth"
    output = tmp_path / "val_predictions.csv"
    np.save(logits, np.array([[0.0, 2.0], [3.0, 1.0]], dtype=np.float32))
    np.save(labels, np.array([1, 0], dtype=np.int64))
    torch.save({"class_to_idx": {"tomato___healthy": 0, "corn___rust": 1}}, checkpoint)
    assert export_predictions(logits, labels, checkpoint, output) == 2
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["predicted_class"] for row in rows] == ["corn___rust", "tomato___healthy"]
    assert [row["correct"] for row in rows] == ["1", "1"]
