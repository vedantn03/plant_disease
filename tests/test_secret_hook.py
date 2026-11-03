"""Regression test: the pinned private-key hook rejects a harmless fake PEM header."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_detect_private_key_hook_blocks_fake_key_pattern() -> None:
    pre_commit = shutil.which("pre-commit")
    if pre_commit is None:
        pytest.skip("pre-commit executable is unavailable")

    fixture = ROOT / "tests" / ".fake_private_key_fixture.pem"
    header = "-----BEGIN " + "RSA PRIVATE KEY-----"
    footer = "-----END " + "RSA PRIVATE KEY-----"
    fixture.write_text(f"{header}\nnot-a-real-key\n{footer}\n", encoding="utf-8")
    try:
        result = subprocess.run(
            [pre_commit, "run", "detect-private-key", "--files", fixture.relative_to(ROOT).as_posix()],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0, result.stdout + result.stderr
        assert "Failed" in result.stdout or "private key" in (result.stdout + result.stderr).lower()
    finally:
        fixture.unlink(missing_ok=True)
