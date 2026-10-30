"""
src/data_validation.py
======================
Automated Data Validation and Quality Assurance Suite (Unit-II Syllabus Requirement).
Implements production-grade assertions on dataset integrity, schema conformity,
and distribution bounds.
"""

from pathlib import Path
from typing import Dict, Any
import json
import pandas as pd
from PIL import Image

from src.utils import get_logger

log = get_logger("data_validation")


class DataValidator:
    """
    Validates image datasets, split manifests, and label distributions.
    Emits auditable JSON validation reports.
    """

    def __init__(self, data_root: Path, splits_dir: Path) -> None:
        self.data_root = Path(data_root)
        self.splits_dir = Path(splits_dir)
        self.results: Dict[str, Any] = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "status": "PASSED",
            "checks_run": 0,
            "checks_passed": 0,
            "checks_failed": 0,
            "details": [],
        }

    def _record_check(self, name: str, passed: bool, message: str, meta: Dict = None) -> None:
        self.results["checks_run"] += 1
        if passed:
            self.results["checks_passed"] += 1
        else:
            self.results["checks_failed"] += 1
            self.results["status"] = "FAILED"
        self.results["details"].append({
            "check": name,
            "passed": passed,
            "message": message,
            "meta": meta or {},
        })
        log_level = log.info if passed else log.error
        log_level(f"[{'PASS' if passed else 'FAIL'}] {name}: {message}")

    def validate_manifests_exist(self) -> bool:
        required_files = ["train.csv", "val.csv", "test.csv"]
        missing = [f for f in required_files if not (self.splits_dir / f).exists()]
        passed = len(missing) == 0
        self._record_check(
            name="manifests_exist",
            passed=passed,
            message="All split manifests exist" if passed else f"Missing: {missing}",
            meta={"splits_dir": str(self.splits_dir)},
        )
        return passed

    def validate_schema_and_nulls(self) -> bool:
        all_passed = True
        required_cols = {"path", "class_name", "crop", "condition", "group_id"}
        for split in ["train.csv", "val.csv", "test.csv"]:
            p = self.splits_dir / split
            if not p.exists():
                continue
            df = pd.read_csv(p)
            missing_cols = required_cols - set(df.columns)
            null_counts = df.isnull().sum().to_dict()
            has_nulls = sum(null_counts.values()) > 0

            passed = (len(missing_cols) == 0) and (not has_nulls)
            if not passed:
                all_passed = False
            self._record_check(
                name=f"schema_and_nulls_{split}",
                passed=passed,
                message=f"{split} has valid columns and zero nulls" if passed else f"Issues in {split}",
                meta={"row_count": len(df), "missing_columns": list(missing_cols), "null_counts": null_counts},
            )
        return all_passed

    def validate_zero_group_leakage(self) -> bool:
        train_p = self.splits_dir / "train.csv"
        test_p = self.splits_dir / "test.csv"
        val_p = self.splits_dir / "val.csv"
        if not (train_p.exists() and test_p.exists()):
            return False

        train_groups = set(pd.read_csv(train_p)["group_id"].unique())
        test_groups = set(pd.read_csv(test_p)["group_id"].unique())
        val_groups = set(pd.read_csv(val_p)["group_id"].unique()) if val_p.exists() else set()

        overlap_train_test = train_groups & test_groups
        overlap_train_val = train_groups & val_groups
        overlap_val_test = val_groups & test_groups

        passed = len(overlap_train_test) == 0 and len(overlap_train_val) == 0 and len(overlap_val_test) == 0
        self._record_check(
            name="zero_group_leakage",
            passed=passed,
            message="Zero group leakage verified across all splits" if passed else "Group overlap detected!",
            meta={
                "train_test_overlap": len(overlap_train_test),
                "train_val_overlap": len(overlap_train_val),
                "val_test_overlap": len(overlap_val_test),
            },
        )
        return passed

    def validate_class_distribution_bounds(self) -> bool:
        train_p = self.splits_dir / "train.csv"
        if not train_p.exists():
            return False
        df = pd.read_csv(train_p)
        class_counts = df["class_name"].value_counts()
        total_classes = len(class_counts)
        min_count = int(class_counts.min())
        max_count = int(class_counts.max())

        # Assert all 38 classes present, min support >= 50, max support <= 6000
        passed = (total_classes == 38) and (min_count >= 50) and (max_count <= 6000)
        self._record_check(
            name="class_distribution_bounds",
            passed=passed,
            message=f"38 classes verified (min_support={min_count}, max_support={max_count})",
            meta={
                "total_classes": total_classes,
                "min_count": min_count,
                "max_count": max_count,
                "imbalance_ratio": round(max_count / max(min_count, 1), 2),
            },
        )
        return passed

    def validate_sample_image_tensors(self, sample_size: int = 50) -> bool:
        train_p = self.splits_dir / "train.csv"
        if not train_p.exists():
            return False
        df = pd.read_csv(train_p).sample(n=min(sample_size, len(pd.read_csv(train_p))), random_state=42)
        corrupted = 0
        invalid_shapes = 0
        invalid_channels = 0

        for _, row in df.iterrows():
            img_path = Path(row["path"])
            if not img_path.exists():
                corrupted += 1
                continue
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    if w < 64 or h < 64:
                        invalid_shapes += 1
                    if img.mode not in ("RGB", "L"):
                        invalid_channels += 1
            except Exception:
                corrupted += 1

        passed = (corrupted == 0) and (invalid_shapes == 0) and (invalid_channels == 0)
        self._record_check(
            name="image_tensor_integrity",
            passed=passed,
            message="All sampled image files valid, readable, and meet resolution bounds" if passed else "Image errors found",
            meta={"samples_checked": len(df), "corrupted": corrupted, "invalid_shapes": invalid_shapes},
        )
        return passed

    def run_all(self, output_report_path: Path = None) -> Dict[str, Any]:
        self.validate_manifests_exist()
        self.validate_schema_and_nulls()
        self.validate_zero_group_leakage()
        self.validate_class_distribution_bounds()
        self.validate_sample_image_tensors()

        if output_report_path:
            output_report_path = Path(output_report_path)
            output_report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_report_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2)
            log.info(f"Data validation report saved to {output_report_path}")

        return self.results
