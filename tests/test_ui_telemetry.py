import json

from fastapi import HTTPException

from src import api


def test_ui_telemetry_returns_bounded_rows_and_aggregates(tmp_path, monkeypatch):
    (tmp_path / "logs").mkdir()
    log = tmp_path / "logs" / "requests.jsonl"
    log.write_text("\n".join([
        json.dumps({"status": "ok", "confidence": 0.8, "latency_ms": 10,
                    "traffic_source": "api"}),
        json.dumps({"status": "uncertain", "confidence": 0.4, "latency_ms": 30,
                    "traffic_source": "replay", "private_field": "synthetic-value"}),
        "not-json",
    ]), encoding="utf-8")
    monkeypatch.setattr(api, "__file__", str(tmp_path / "src" / "api.py"))

    result = api.ui_telemetry(limit=1)

    assert result["available"] is True
    assert result["record_count"] == 1
    assert result["records"][0]["status"] == "uncertain"
    assert result["summary"]["mean_latency_ms"] == 30
    assert result["summary"]["traffic_sources"] == {"replay": 1}
    assert result["malformed_lines"] == 1
    assert "private_field" not in result["records"][0]


def test_ui_telemetry_returns_clear_empty_state(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "__file__", str(tmp_path / "src" / "api.py"))
    result = api.ui_telemetry(limit=100)
    assert result["available"] is False
    assert result["records"] == []
    assert "No request log" in result["message"]


def test_ui_telemetry_omits_non_finite_metrics(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "requests.jsonl").write_text(
        '{"confidence": NaN, "latency_ms": Infinity, "status": "ok"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "__file__", str(tmp_path / "src" / "api.py"))

    result = api.ui_telemetry(limit=100)

    assert result["records"][0]["confidence"] is None
    assert result["records"][0]["latency_ms"] is None
    assert result["summary"]["mean_confidence"] is None
    assert result["summary"]["mean_latency_ms"] is None


def test_ui_report_rejects_unlisted_paths():
    try:
        api.resolve_ui_report("../.env")
    except HTTPException as error:
        assert error.status_code == 404
    else:
        raise AssertionError("Unlisted report paths must not be resolved")


def test_ui_pipeline_reports_declared_stages_without_reproduction(tmp_path, monkeypatch):
    (tmp_path / "dvc.yaml").write_text(
        "stages:\n  audit:\n    cmd: python audit.py\n  evaluate:\n    cmd: python evaluate.py\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "__file__", str(tmp_path / "src" / "api.py"))

    result = api.ui_pipeline()

    assert result == {
        "stages": ["audit", "evaluate"],
        "source": "dvc.yaml",
        "reproduction_run": False,
    }


def test_specific_ui_data_routes_precede_report_name_route():
    paths = [route.path for route in api.app.routes]
    report_route = paths.index("/ui-data/{report_name}")
    assert paths.index("/ui-data/telemetry") < report_route
    assert paths.index("/ui-data/pipeline") < report_route
