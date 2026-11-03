"""Local-only admin reload checks without loading model weights."""

import pytest
from fastapi import HTTPException
from starlette.requests import Request

import src.api as api
from src.api import _authorize_admin_reload, _is_local_admin_request


def _request(client_host: str, host_header: str = "127.0.0.1:8000") -> Request:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/admin/reload",
        "raw_path": b"/admin/reload",
        "query_string": b"",
        "headers": [(b"host", host_header.encode("ascii"))],
        "client": (client_host, 12345),
        "server": ("127.0.0.1", 8000),
    }
    return Request(scope)


def test_admin_reload_requires_loopback_client_and_local_host():
    assert _is_local_admin_request(_request("127.0.0.1")) is True
    assert _is_local_admin_request(_request("172.20.0.1")) is True
    assert _is_local_admin_request(_request("8.8.8.8")) is False
    assert _is_local_admin_request(_request("127.0.0.1", "remote.example")) is False


def test_admin_reload_requires_token(monkeypatch):
    monkeypatch.setenv("ADMIN_RELOAD_TOKEN", "local-test-token")
    with pytest.raises(HTTPException) as denied:
        _authorize_admin_reload(_request("127.0.0.1"), "wrong-token")
    assert denied.value.status_code == 401
    _authorize_admin_reload(_request("127.0.0.1"), "local-test-token")
    with pytest.raises(HTTPException) as remote:
        _authorize_admin_reload(_request("8.8.8.8"), "local-test-token")
    assert remote.value.status_code == 403


def test_admin_reload_requires_configured_token(monkeypatch):
    monkeypatch.delenv("ADMIN_RELOAD_TOKEN", raising=False)
    with pytest.raises(HTTPException) as missing:
        _authorize_admin_reload(_request("127.0.0.1"), "token")
    assert missing.value.status_code == 503


def test_model_info_gauge_changes_to_reloaded_version(monkeypatch):
    class _Gauge:
        def __init__(self):
            self.removed = []
            self.values = []

        def remove(self, **labels):
            self.removed.append(labels)

        def labels(self, **labels):
            self.values.append(labels)
            return self

        def set(self, value):
            self.value = value

    gauge = _Gauge()
    monkeypatch.setattr(api, "MODEL_INFO", gauge)
    monkeypatch.setattr(api, "_state", {
        "grad_cam": None,
        "model_version": "2",
        "model_provenance": {"model_source": "registry"},
    })
    api._install_model_bundle({
        "model": object(),
        "grad_cam": None,
        "model_version": "3",
        "model_provenance": {"model_source": "registry"},
    })
    assert gauge.removed == [{"version": "2", "model_source": "registry"}]
    assert gauge.values == [{"version": "3", "model_source": "registry"}]
    assert gauge.value == 1
