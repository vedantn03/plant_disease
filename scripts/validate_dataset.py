"""
scripts/validate_dataset.py
===========================
CLI execution runner for the Data Validation Suite (Unit-II).
Outputs reports/data_validation_report.json.
"""

from pathlib import Path
import sys

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_validation import DataValidator
from src.utils import get_logger

log = get_logger("validate_dataset")

def main():
    splits_dir = PROJECT_ROOT / "data" / "splits"
    data_root = PROJECT_ROOT
    report_path = PROJECT_ROOT / "reports" / "data_validation_report.json"

    log.info("Starting automated Data Validation Suite...")
    validator = DataValidator(data_root=data_root, splits_dir=splits_dir)
    results = validator.run_all(output_report_path=report_path)

    log.info(f"Validation status: {results['status']} ({results['checks_passed']}/{results['checks_run']} checks passed)")
    if results['status'] != "PASSED":
        sys.exit(1)

if __name__ == "__main__":
    main()
