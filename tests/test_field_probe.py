"""Field-probe report tests use temporary, synthetic image files."""

import json

from PIL import Image

from scripts.field_probe import run_field_probe


class _FakeResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        return None

    def json(self):
        return dict(self.body)


def test_field_probe_summarizes_temp_images_without_ground_truth(tmp_path):
    image_dir = tmp_path / "field_probe"
    image_dir.mkdir()
    Image.new("RGB", (8, 8), (20, 120, 30)).save(image_dir / "one.jpg")
    Image.new("RGB", (8, 8), (120, 120, 120)).save(image_dir / "two.png")
    (image_dir / "notes.txt").write_text("not an image", encoding="utf-8")
    bodies = iter([
        {
            "status": "ok",
            "crop": "apple",
            "condition": "healthy",
            "confidence": 0.8,
            "bg_check": "consistent",
        },
        {
            "status": "rejected_not_a_leaf",
            "reason_code": "not_leaf",
            "bg_check": "unknown",
        },
    ])

    def fake_post(*_args, **_kwargs):
        return _FakeResponse(next(bodies))

    output_path = tmp_path / "field_probe.json"
    report = run_field_probe(
        image_dir,
        api_url="http://api.test",
        timeout_seconds=1,
        report_path=output_path,
        post=fake_post,
    )
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["title"] == "indicative only; no ground truth"
    assert report["label_status"] == "UNLABELLED"
    assert report["image_count"] == 2
    assert report["status_counts"] == {"ok": 1, "rejected_not_a_leaf": 1}
    assert report["abstain_reason_counts"] == {"not_leaf": 1}
    assert report["bg_check_counts"] == {"consistent": 1, "inconsistent": 0, "unknown": 1}
    assert report["top_predicted_classes"] == [{"class": "apple___healthy", "count": 1}]
    assert saved["title"] == report["title"]
    assert "responses" not in saved
    assert "filename" not in saved
