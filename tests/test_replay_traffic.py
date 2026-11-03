"""Replay traffic CLI and degradation helper tests use synthetic images only."""

from PIL import Image
import pytest

from scripts.replay_traffic import degrade_image, load_test_paths


@pytest.mark.parametrize("mode", ["blur", "dark", "noise"])
def test_degradation_functions_return_rgb_images(mode):
    image = Image.new("RGB", (32, 32), (120, 130, 140))
    degraded = degrade_image(image, mode)
    assert degraded.mode == "RGB"
    assert degraded.size == image.size


def test_invalid_degradation_is_rejected():
    with pytest.raises(ValueError, match="Unsupported degradation"):
        degrade_image(Image.new("RGB", (2, 2)), "invalid")


def test_replay_cli_arguments_and_seeded_selection(tmp_path):
    manifest = tmp_path / "test.csv"
    manifest.write_text("path,class_name\na.jpg,A___healthy\nb.jpg,B___healthy\n", encoding="utf-8")
    assert len(load_test_paths(manifest, limit=1, seed=42)) == 1


def test_replay_cli_rejects_invalid_rate(monkeypatch):
    import sys
    from scripts import replay_traffic

    monkeypatch.setattr(sys, "argv", ["replay_traffic.py", "--rate", "0"])
    with pytest.raises(SystemExit):
        replay_traffic.main()
