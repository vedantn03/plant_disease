"""
scripts/run_airflow_simulation.py
=================================
Local execution harness simulating the Apache Airflow DAG (Unit-III).
Executes the pipeline tasks in topological dependency order and logs status.
Outputs:
  - reports/airflow_dag_run.json
"""

from pathlib import Path
import sys
import time
import json
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_logger

log = get_logger("airflow_runner")


def main():
    log.info("=" * 65)
    log.info("Apache Airflow DAG Execution Simulator: plant_disease_retraining")
    log.info("DAG Schedule: 0 0 * * 0 (Weekly) | Trigger: Data Drift Sensor")
    log.info("=" * 65)

    dag_tasks = [
        {
            "task_id": "check_drift_trigger",
            "operator": "PythonOperator",
            "command": "python scripts/check_drift_and_trigger.py",
            "description": "Inspects reports/drift_report.json for KS / PSI distribution shift",
        },
        {
            "task_id": "data_audit_and_split",
            "operator": "BashOperator",
            "command": "python scripts/validate_dataset.py",
            "description": "Verifies zero-leakage group splits, SHA-256 integrity, and manifests",
        },
        {
            "task_id": "data_validation_suite",
            "operator": "PythonOperator",
            "command": "python scripts/validate_dataset.py",
            "description": "Validates schema, null checks, group bounds, and tensor integrity",
        },
        {
            "task_id": "evaluate_ood_detection",
            "operator": "BashOperator",
            "command": "python scripts/evaluate_ood.py",
            "description": "Evaluates Helmholtz Free Energy Score and Layer 1 Heuristics on OOD inputs",
        },
        {
            "task_id": "benchmark_xai_explainability",
            "operator": "BashOperator",
            "command": "python scripts/benchmark_xai.py",
            "description": "Executes Grad-CAM, SHAP, and LIME attribution mass comparative benchmark",
        },
        {
            "task_id": "evaluate_promotion_gate",
            "operator": "BashOperator",
            "command": "python scripts/promote.py",
            "description": "Evaluates 4-criterion gate (Macro-F1, Recall Floor, AUROC, Background Retention)",
        },
    ]

    dag_run_log = {
        "dag_id": "plant_disease_continuous_retraining",
        "run_id": f"manual_sim__{int(time.time())}",
        "execution_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "SUCCESS",
        "tasks": [],
    }

    all_success = True
    python_exe = sys.executable

    for task in dag_tasks:
        t0 = time.perf_counter()
        log.info(f"Executing Airflow Task [{task['task_id']}] ({task['operator']})...")
        cmd_args = task["command"].split()
        cmd_args[0] = python_exe

        try:
            res = subprocess.run(cmd_args, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=120)
            elapsed = time.perf_counter() - t0
            task_status = "SUCCESS" if res.returncode == 0 else "FAILED"
            if res.returncode != 0:
                all_success = False

            log.info(f"  -> Task [{task['task_id']}] finished in {elapsed:.2f}s with status: {task_status}")
            dag_run_log["tasks"].append({
                "task_id": task["task_id"],
                "operator": task["operator"],
                "status": task_status,
                "duration_seconds": round(elapsed, 2),
                "exit_code": res.returncode,
                "description": task["description"],
            })
        except Exception as e:
            elapsed = time.perf_counter() - t0
            log.error(f"  -> Task [{task['task_id']}] encountered exception: {e}")
            all_success = False
            dag_run_log["tasks"].append({
                "task_id": task["task_id"],
                "operator": task["operator"],
                "status": "FAILED",
                "duration_seconds": round(elapsed, 2),
                "error": str(e),
                "description": task["description"],
            })

    dag_run_log["status"] = "SUCCESS" if all_success else "FAILED"
    out_path = PROJECT_ROOT / "reports" / "airflow_dag_run.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dag_run_log, f, indent=2)

    log.info(f"Airflow DAG simulation complete. State logged to {out_path}")


if __name__ == "__main__":
    main()
