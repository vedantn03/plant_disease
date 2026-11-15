"""Static Compose and Grafana checks; do not require a Docker daemon."""

import json
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class _DashboardMarkup(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.internal_links = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            self.ids.append(values["id"])
        if tag == "a" and values.get("href", "").startswith("#"):
            self.internal_links.append(values["href"][1:])


def _yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_ci_pillow_range_is_compatible_with_pinned_streamlit():
    requirements = (ROOT / "requirements-ci.txt").read_text(encoding="utf-8")
    assert re.search(r"(?im)^pillow\s*>=\s*7\.1\.0\s*,\s*<\s*12\s*$", requirements)
    assert re.search(r"(?im)^streamlit==1\.45\.1\s*$", requirements)


def test_ui_is_static_web_frontend_with_local_api_origin():
    compose = _yaml(ROOT / "docker-compose.yml")
    ui = compose["services"]["ui"]
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "web/server.py" in dockerfile
    assert "depends_on" not in ui
    assert "LEAFWISE_API_URL" in (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    insights_html = (ROOT / "web" / "insights.html").read_text(encoding="utf-8")
    insights_script = (ROOT / "web" / "insights.js").read_text(encoding="utf-8")
    css = (ROOT / "web" / "style.css").read_text(encoding="utf-8")
    assert 'id="upload-form"' in html
    assert 'api("/predict"' in script
    assert "payload.advice_trace" in script
    assert 'id="demo-sample"' in html
    assert "demo_samples.json" in script
    assert "demo_samples/*.jpg" in (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert 'href="/insights"' in html
    assert 'capture="environment"' in html
    assert 'id="temperature"' in html and 'id="humidity"' in html
    assert 'role="tab"' in insights_html
    assert '"/explain/jobs"' in script
    assert 'get("/insights")' in insights_script
    assert 'id="disclaimer"' in html
    assert 'option value="hi"' not in html
    assert 'option value="mr"' in html
    assert '"disclaimer"' in (ROOT / "web" / "explanation_strings.json").read_text(encoding="utf-8")
    assert "payload.confusable_hints" in script
    assert "item.label" in script
    assert "weather_source_citation" in script
    assert "@media(max-width:520px){#submit{position:sticky" in css.replace(" ", "")
    assert "[hidden]{display:none!important}" in css.replace(" ", "")
    assert "@import url('https://" not in css
    assert "FROM python:3.12-slim AS ui" in dockerfile
    assert '"lime"' not in str(ui).lower()
    assert '"shap"' not in str(ui).lower()
    assert "@import url('https://" not in css
    ui_sources = html + script + insights_html + insights_script
    assert not re.search(r"https?://(?:cdnjs|unpkg|fonts\.googleapis|cdn\.)", ui_sources)
    assert 'id="mlflow-run"' in insights_html
    assert "allowlisted parameters" in insights_script
    markup = _DashboardMarkup()
    markup.feed(html)
    assert len(markup.ids) == len(set(markup.ids))
    assert set(markup.internal_links) <= set(markup.ids)


def test_browser_sample_manifest_omits_machine_paths_and_files_exist():
    samples = json.loads((ROOT / "web" / "demo_samples.json").read_text(encoding="utf-8"))
    source_manifest = (ROOT / "demo_samples" / "manifest.json").read_text(encoding="utf-8")
    assert samples
    assert all(set(item) == {"filename", "label"} for item in samples)
    assert all("source_path" not in item for item in samples)
    assert all((ROOT / "demo_samples" / item["filename"]).is_file() for item in samples)
    assert '"source_path"' not in source_manifest
    assert not re.search(r"[A-Za-z]:\\", source_manifest)


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

    assert "${MODEL_DIR:-./models}:/app/models:ro" in services["api"]["volumes"]
    assert "${DATA_DIR:-./data}:/app/data:ro" in services["api"]["volumes"]
    assert "${REPORTS_DIR:-./reports}:/app/reports:ro" in services["api"]["volumes"]
    assert "${MLFLOW_DB_PATH:-./mlflow.db}:/app/mlflow.db:ro" in services["api"]["volumes"]
    assert services["api"].get("environment", {}).get("DATA_ROOT") == "/dataset"
    assert "${DATA_ROOT:-./data}:/dataset:ro" in services["api"]["volumes"]
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
