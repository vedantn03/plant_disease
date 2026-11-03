"""Tests for the repository artifact policy using temporary tracked-list fixtures."""

from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_tracked_files.py"
SPEC = importlib.util.spec_from_file_location("check_tracked_files", MODULE_PATH)
assert SPEC and SPEC.loader
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def test_configured_limit_matches_precommit_hook() -> None:
    config = checker.configured_max_bytes()
    assert config == 5120 * 1024


def test_check_detects_prohibited_artifacts_and_large_files(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    env_file = tmp_path / ".env"
    env_local = tmp_path / ".env.local"
    env_example = tmp_path / ".env.example"
    model = tmp_path / "checkpoint.pth"
    database = tmp_path / "mlflow.db"
    large = tmp_path / "large.bin"
    for path in (env_file, env_local, env_example, model, database, large):
        path.write_bytes(b"x")

    monkeypatch.setattr(checker, "ROOT", tmp_path)
    monkeypatch.setattr(
        checker,
        "tracked_paths",
        lambda: [env_file, env_local, env_example, model, database, large],
    )
    monkeypatch.setattr(checker, "configured_max_bytes", lambda: 0)

    assert checker.main() == 1
    output = capsys.readouterr().out
    assert "prohibited tracked file: .env" in output
    assert "prohibited tracked file: .env.local" in output
    assert "prohibited tracked file: .env.example" not in output
