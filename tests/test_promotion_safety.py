"""Checkpoint DVC validation tests use temporary synthetic bytes only."""

import hashlib
import json

import yaml

from src.promotion_safety import (
    _candidate_directory_output_status,
    checkpoint_dvc_status,
)


def test_dvc_pointer_must_match_checkpoint_bytes(tmp_path):
    checkpoint = tmp_path / "candidate.pth"
    checkpoint.write_bytes(b"synthetic checkpoint bytes")
    digest = hashlib.md5(checkpoint.read_bytes(), usedforsecurity=False).hexdigest()
    pointer = checkpoint.with_name("candidate.pth.dvc")
    pointer.write_text(yaml.safe_dump({"outs": [{"md5": f"{digest}.md5"}]}), encoding="utf-8")

    valid, reason = checkpoint_dvc_status(
        checkpoint, git_root=tmp_path, require_git_tracked=False
    )
    assert valid is True
    assert "matches" in reason

    checkpoint.write_bytes(b"different synthetic bytes")
    valid, reason = checkpoint_dvc_status(
        checkpoint, git_root=tmp_path, require_git_tracked=False
    )
    assert valid is False
    assert "does not match" in reason


def test_missing_dvc_pointer_is_rejected(tmp_path):
    checkpoint = tmp_path / "candidate.pth"
    checkpoint.write_bytes(b"synthetic checkpoint bytes")
    valid, reason = checkpoint_dvc_status(
        checkpoint, git_root=tmp_path, require_git_tracked=False
    )
    assert valid is False
    assert "Individual DVC pointer is absent" in reason


def test_candidate_in_train_directory_output_is_verified_from_dvc_lock_and_cache(tmp_path):
    candidate = tmp_path / "models" / "candidates" / "run-x.pth"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"synthetic candidate checkpoint")
    candidate_hash = hashlib.md5(candidate.read_bytes(), usedforsecurity=False).hexdigest()
    directory_hash = hashlib.md5(b"synthetic DVC directory manifest", usedforsecurity=False).hexdigest()
    lock = tmp_path / "dvc.lock"
    lock.write_text(yaml.safe_dump({
        "stages": {"train": {"outs": [{
            "path": "models/candidates",
            "md5": f"{directory_hash}.dir",
        }]}}
    }), encoding="utf-8")
    cache = tmp_path / "cache"
    dvc_files = cache / "files" / "md5"
    directory_object = dvc_files / directory_hash[:2] / directory_hash[2:]
    directory_object.parent.mkdir(parents=True)
    directory_object.write_text(json.dumps([{
        "md5": candidate_hash,
        "relpath": "run-x.pth",
    }]), encoding="utf-8")
    candidate_object = dvc_files / candidate_hash[:2] / candidate_hash[2:]
    candidate_object.parent.mkdir(parents=True)
    candidate_object.write_bytes(candidate.read_bytes())

    valid, reason = _candidate_directory_output_status(
        candidate,
        git_root=tmp_path,
        require_git_tracked=False,
        cache_dir=cache,
    )
    assert valid is True
    assert "committed DVC train directory output" in reason
