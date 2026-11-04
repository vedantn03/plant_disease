from __future__ import annotations

import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_architecture_diagram_only_names_compose_services() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    text = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    diagram_services = set(re.findall(r"\bcompose_([a-z][a-z0-9-]*)\s*\[", text))
    assert diagram_services
    assert diagram_services <= set(compose["services"])
