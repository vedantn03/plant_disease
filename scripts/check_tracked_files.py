"""Fail CI if prohibited local artifacts or oversized files are tracked."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def configured_max_bytes() -> int:
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    for repo in config["repos"]:
        for hook in repo.get("hooks", []):
            if hook["id"] == "check-added-large-files":
                for arg in hook.get("args", []):
                    if arg.startswith("--maxkb="):
                        return int(arg.split("=", 1)[1]) * 1024
    raise ValueError("check-added-large-files must define --maxkb in pre-commit config")


def tracked_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    )
    return [ROOT / path.decode() for path in result.stdout.split(b"\0") if path]


def main() -> int:
    max_bytes = configured_max_bytes()
    failures: list[str] = []
    for path in tracked_paths():
        relative = path.relative_to(ROOT).as_posix()
        name = path.name.lower()
        if (
            name == ".env"
            or (name.startswith(".env.") and name != ".env.example")
            or name.endswith((".pth", ".db"))
        ):
            failures.append(f"prohibited tracked file: {relative}")
        if path.is_file() and path.stat().st_size > max_bytes:
            failures.append(
                f"tracked file exceeds configured size limit ({max_bytes} bytes): {relative}"
            )
    if failures:
        print("\n".join(failures))
        return 1
    print(f"Tracked-file check passed (configured limit: {max_bytes} bytes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
