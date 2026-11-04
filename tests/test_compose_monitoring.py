"""Static Compose and Grafana checks; do not require a Docker daemon."""

import json
import re
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_compose_profiles_healthchecks_and_read_only_model_mount():
    compose = _yaml(ROOT / "docker-compose.yml")
    services = compose["services"]
    assert set(services["api"]["profiles"]) == {"core"}
    assert set(services["ui"]["profiles"]) == {"core"}
    assert set(services["prometheus"]["profiles"]) == {"monitoring"}
    assert set(services["grafana"]["profiles"]) == {"monitoring"}
    assert set(services["drift-exporter"]["profiles"]) == {"monitoring"}

    for service in services.values():
        assert service.get("healthcheck", {}).get("test")
        assert service.get("restart") == "unless-stopped"

    assert "./models:/app/models:ro" in services["api"]["volumes"]
    assert "./reports:/app/reports:ro" in services["api"]["volumes"]
    assert "DATA_ROOT" not in services["api"].get("environment", {})
    assert {"prometheus_data", "grafana_data"} <= set(compose["volumes"])
    ui_build = services["ui"].get("build", {})
    assert ui_build.get("target") == "ui"
    assert "lime" not in str(services["ui"]).lower()
    assert "shap" not in str(services["ui"]).lower()
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8").lower()
    assert not re.search(r"(?m)^\s*(lime|shap)\s*(?:\\)?$", dockerfile)
    ci_requirements = (ROOT / "requirements-ci.txt").read_text(encoding="utf-8")
    assert "streamlit==1.45.1" in ci_requirements


def test_prometheus_config_scrapes_api_and_has_empty_rules():
    config = _yaml(ROOT / "monitoring" / "prometheus.yml")
    assert config["global"]["scrape_interval"] == "15s"
    assert config["rule_files"] == ["rules/drift.yml"]
    job = config["scrape_configs"][0]
    assert job["metrics_path"] == "/metrics"
    assert job["static_configs"][0]["targets"] == ["api:8000"]
    drift_job = config["scrape_configs"][1]
    assert drift_job["static_configs"][0]["targets"] == ["drift-exporter:9101"]


def test_prometheus_drift_rule_duration_matches_config():
    rule_data = _yaml(ROOT / "monitoring" / "rules" / "drift.yml")
    config = _yaml(ROOT / "configs" / "config.yaml")
    rule = rule_data["groups"][0]["rules"][0]
    assert rule["expr"] == "drift_flag == 1"
    assert rule["for"] == config["drift"]["alert_for"]


def test_dashboard_queries_only_metrics_declared_by_api():
    dashboard_path = (
        ROOT / "monitoring" / "grafana" / "dashboards" / "system-health-model-trust.json"
    )
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    assert dashboard["title"] == "System health and model trust"
    expressions = [
        target["expr"]
        for panel in dashboard["panels"]
        for target in panel.get("targets", [])
    ]

    source_text = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in ("src/api.py", "scripts/drift_exporter.py")
    )
    declared = set(re.findall(r"(?:Counter|Gauge|Histogram)\(\s*['\"]([^'\"]+)", source_text))
    derived = declared | {f"{name}_bucket" for name in declared}
    known_functions_and_labels = {
        "sum", "rate", "histogram_quantile", "increase", "by",
        "status", "traffic_source", "reason", "le",
        "tier", "evidence_grade", "class",
    }
    tokens = {
        token
        for expression in expressions
        for token in re.findall(r"\b[a-zA-Z_:][a-zA-Z0-9_:]*\b", expression)
        if "_" in token or ":" in token
    }
    references = tokens - known_functions_and_labels

    assert references
    assert references <= derived
    assert all("vector(" not in expression for expression in expressions)
    assert any("predict_requests_total" in expression for expression in expressions)
    assert any("predict_latency_seconds_bucket" in expression for expression in expressions)
    assert any("predict_confidence_bucket" in expression for expression in expressions)
    assert any("predict_abstain_total" in expression for expression in expressions)
    assert any("model_info" in expression for expression in expressions)
    assert any("bg_check_total" in expression for expression in expressions)
    assert any("advice_total" in expression for expression in expressions)
    assert any("advice_withheld_total" in expression for expression in expressions)
    assert any("advice_gap_total" in expression for expression in expressions)
    assert any(panel["title"] == "Advice coverage" for panel in dashboard["panels"])


def test_grafana_provisioning_and_credentials_have_no_committed_secret():
    datasource = _yaml(
        ROOT / "monitoring" / "grafana" / "provisioning" / "datasources" / "prometheus.yml"
    )
    provider = _yaml(
        ROOT / "monitoring" / "grafana" / "provisioning" / "dashboards" / "dashboards.yml"
    )
    assert datasource["datasources"][0]["url"] == "http://prometheus:9090"
    assert provider["providers"][0]["options"]["path"] == "/var/lib/grafana/dashboards"

    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "GRAFANA_ADMIN_USER=replace-this-user" in example
    assert "GRAFANA_ADMIN_PASSWORD=replace-with-a-local-secret" in example
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore.splitlines()
    tracked_env = subprocess.run(
        ["git", "ls-files", "--error-unmatch", ".env"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert tracked_env.returncode != 0
