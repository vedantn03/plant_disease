"""
tests/test_dvc_pipeline.py
==========================
Structural integrity checks for the DVC pipeline definition.

Rules (from AGENTS.md):
  - Every stage's script must exist under scripts/ or src/.
  - Every params key referenced in dvc.yaml must exist in configs/config.yaml.
  - The dvc.yaml file must be parseable YAML.
  - Every stage must declare at least one dep and one out.

These tests are fast (no I/O other than YAML parsing) and run in < 1 s.
"""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DVC_YAML_PATH = PROJECT_ROOT / "dvc.yaml"
CONFIG_YAML_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def _load_dvc_yaml() -> dict:
    with open(DVC_YAML_PATH) as f:
        return yaml.safe_load(f)


def _flatten_config_keys(obj: dict, prefix: str = "") -> set[str]:
    """Return every dotted key path present in the config dict."""
    keys: set[str] = set()
    for k, v in obj.items():
        full_key = f"{prefix}.{k}" if prefix else k
        keys.add(full_key)
        if isinstance(v, dict):
            keys.update(_flatten_config_keys(v, prefix=full_key))
    return keys


def test_dvc_yaml_is_parseable():
    """dvc.yaml must be valid YAML and contain a top-level 'stages' key."""
    data = _load_dvc_yaml()
    assert "stages" in data, "dvc.yaml must have a top-level 'stages' key"
    assert isinstance(data["stages"], dict), "'stages' must be a mapping"


def test_every_stage_script_exists():
    """The script referenced in every stage's 'cmd' must exist on disk."""
    data = _load_dvc_yaml()
    missing = []
    for stage_name, stage in data["stages"].items():
        cmd = stage.get("cmd", "")
        # The cmd is always: python <path/to/script.py> [args]
        parts = cmd.strip().split()
        if len(parts) >= 2 and parts[0] == "python":
            script_path = PROJECT_ROOT / parts[1]
            if not script_path.exists():
                missing.append(f"{stage_name}: {parts[1]!r} not found")
    assert not missing, "Missing stage scripts:\n" + "\n".join(missing)


def test_every_stage_has_deps_and_outs():
    """Each stage must declare at least one dep and at least one out."""
    data = _load_dvc_yaml()
    problems = []
    for stage_name, stage in data["stages"].items():
        deps = stage.get("deps", [])
        outs = stage.get("outs", [])
        if not deps:
            problems.append(f"{stage_name}: no deps declared")
        if not outs:
            problems.append(f"{stage_name}: no outs declared")
    assert not problems, "Stages with missing deps/outs:\n" + "\n".join(problems)


def test_every_params_key_exists_in_config():
    """
    Every params key referenced in dvc.yaml must exist in configs/config.yaml.

    dvc.yaml params block may be written in two forms:
      - configs/config.yaml:
          - seed
          - model.arch
    or as a flat list:
      - seed

    This test handles both forms.
    """
    with open(CONFIG_YAML_PATH) as f:
        raw_config = yaml.safe_load(f)

    config_keys = _flatten_config_keys(raw_config)
    data = _load_dvc_yaml()

    missing_params = []
    for stage_name, stage in data["stages"].items():
        params_block = stage.get("params", [])
        for entry in params_block:
            if isinstance(entry, dict):
                # {config_file: [key1, key2, ...]}
                for _file, keys in entry.items():
                    if keys is None:
                        continue
                    for key in keys:
                        if key not in config_keys:
                            missing_params.append(f"{stage_name}: '{key}' not in config.yaml")
            elif isinstance(entry, str):
                if entry not in config_keys:
                    missing_params.append(f"{stage_name}: '{entry}' not in config.yaml")

    assert not missing_params, (
        "Params keys in dvc.yaml that are absent from configs/config.yaml:\n"
        + "\n".join(missing_params)
    )


def test_no_double_tracked_files():
    """
    Files that are git-tracked (listed in data/splits/manifest_hashes.json by name)
    must not also appear as DVC outs. Double-tracking causes 'dvc commit' to fail.

    We check the known double-tracking case: manifest_hashes.json.
    """
    data = _load_dvc_yaml()
    forbidden = {"data/splits/manifest_hashes.json"}
    violations = []
    for stage_name, stage in data["stages"].items():
        outs = stage.get("outs", [])
        for out in outs:
            out_str = out if isinstance(out, str) else list(out.keys())[0]
            if out_str in forbidden:
                violations.append(f"{stage_name}: '{out_str}' is git-tracked but also a DVC out")
    assert not violations, "Double-tracked files found:\n" + "\n".join(violations)


def test_promotion_stage_is_dry_run_only():
    """The DVC promotion stage must never move the champion alias."""
    data = _load_dvc_yaml()
    promote_cmd = data["stages"]["promote"]["cmd"]
    assert "--dry-run" in promote_cmd
    assert "--apply" not in promote_cmd


def test_candidate_training_path_never_targets_champion_checkpoint():
    """Candidate training must write below models/candidates only."""
    source = (PROJECT_ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    assert "candidate_checkpoint = candidates_dir / f\"{run_id}.pth\"" in source
    assert "models/best_model.pth" not in source
