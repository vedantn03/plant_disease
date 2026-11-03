"""Local checkpoint safety checks used before registry alias changes."""

from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path
import yaml

from src.model_loader import file_md5


def dvc_pointer_for(checkpoint_path: Path) -> Path:
    return Path(f"{checkpoint_path}.dvc")


def checkpoint_dvc_status(
    checkpoint_path: Path,
    *,
    git_root: Path,
    require_git_tracked: bool = True,
) -> tuple[bool, str]:
    """Verify checkpoint membership in an individual DVC pointer or tracked train output."""
    if not checkpoint_path.is_file():
        return False, f"Candidate checkpoint does not exist: {checkpoint_path}"
    pointer_path = dvc_pointer_for(checkpoint_path)
    target_hash = file_md5(checkpoint_path)

    if pointer_path.is_file():
        try:
            pointer = yaml.safe_load(pointer_path.read_text(encoding="utf-8")) or {}
            outputs = pointer.get("outs", [])
            if any(
                str(output.get("md5", "")).removesuffix(".md5").lower() == target_hash
                for output in outputs
            ):
                return _require_committed_file(
                    pointer_path, git_root, require_git_tracked,
                    success_message="Individual DVC pointer matches the checkpoint",
                )
        except (OSError, AttributeError, TypeError, yaml.YAMLError) as exc:
            pointer_error = f"Could not validate individual DVC pointer: {exc}"
        else:
            pointer_error = "Individual DVC pointer MD5 does not match the candidate checkpoint"
    else:
        pointer_error = f"Individual DVC pointer is absent: {pointer_path}"

    directory_status = _candidate_directory_output_status(
        checkpoint_path, git_root=git_root, require_git_tracked=require_git_tracked
    )
    if directory_status[0]:
        return directory_status
    return False, f"{pointer_error}; {directory_status[1]}"


def _require_committed_file(
    path: Path,
    git_root: Path,
    require_git_tracked: bool,
    *,
    success_message: str,
) -> tuple[bool, str]:
    if not require_git_tracked:
        return True, success_message
    try:
        relative_path = path.resolve().relative_to(git_root.resolve()).as_posix()
    except ValueError:
        return False, f"DVC metadata is outside the Git project: {path}"
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative_path],
        cwd=git_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode != 0:
        return False, f"DVC metadata must be tracked: {relative_path}"
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", relative_path],
        cwd=git_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if clean.returncode != 0:
        return False, f"DVC metadata must be committed before apply: {relative_path}"
    return True, success_message


def _candidate_directory_output_status(
    checkpoint_path: Path,
    *,
    git_root: Path,
    require_git_tracked: bool,
    cache_dir: Path | None = None,
) -> tuple[bool, str]:
    """Check a candidate file inside DVC's locked ``models/candidates`` directory output."""
    stage_output = (git_root / "models" / "candidates").resolve()
    checkpoint_path = checkpoint_path.resolve()
    try:
        relative_candidate = checkpoint_path.relative_to(stage_output).as_posix()
    except ValueError:
        return False, "Checkpoint is outside the DVC train output models/candidates"

    lock_path = git_root / "dvc.lock"
    try:
        lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
        train_stage = lock.get("stages", {}).get("train", {})
        output = next(
            item for item in train_stage.get("outs", [])
            if Path(str(item.get("path", ""))).as_posix() == "models/candidates"
        )
        directory_hash = str(output.get("md5", "")).removesuffix(".dir").lower()
        if len(directory_hash) != 32:
            return False, "dvc.lock has no valid hash for the train candidates directory"
    except (OSError, AttributeError, TypeError, StopIteration, yaml.YAMLError) as exc:
        return False, f"Could not read train candidate output from dvc.lock: {exc}"

    if cache_dir is None:
        dvc_executable = Path(sys.executable).with_name("dvc.exe" if sys.platform == "win32" else "dvc")
        if not dvc_executable.is_file():
            return False, "Cannot locate the environment's DVC executable to inspect its cache"
        result = subprocess.run(
            [str(dvc_executable), "cache", "dir"],
            cwd=git_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return False, f"Cannot determine the DVC cache directory: {result.stderr.strip()}"
        cache_dir = Path(result.stdout.strip()).expanduser()
        if not cache_dir.is_absolute():
            cache_dir = git_root / cache_dir

    cache_object = cache_dir / "files" / "md5" / directory_hash[:2] / directory_hash[2:]
    try:
        entries = json.loads(cache_object.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"DVC candidates directory cache entry is unavailable: {exc}"
    matching = next(
        (item for item in entries if item.get("relpath") == relative_candidate), None
    )
    if matching is None:
        return False, f"Candidate is absent from the DVC directory manifest: {relative_candidate}"
    expected_hash = str(matching.get("md5", "")).removesuffix(".md5").lower()
    if expected_hash != file_md5(checkpoint_path):
        return False, "DVC directory manifest MD5 does not match the candidate checkpoint"
    file_object = cache_dir / "files" / "md5" / expected_hash[:2] / expected_hash[2:]
    if not file_object.is_file():
        return False, "Candidate checkpoint is listed in DVC metadata but its cache object is missing"

    committed, message = _require_committed_file(
        lock_path,
        git_root,
        require_git_tracked,
        success_message="Candidate matches the committed DVC train directory output",
    )
    return committed, message
