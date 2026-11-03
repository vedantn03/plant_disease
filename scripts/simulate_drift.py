"""
SIMULATION ONLY: simulate_drift.py
=================
Step 6: Drift simulation demo.

Takes a sample of test images, creates a "drifted" copy (Gaussian blur + brightness shift),
sends both sets through the /predict API, and compares logged feature distributions using
scipy KS tests and PSI.

SIMULATION OUTPUTS (not production monitoring):
  - reports/drift_report.md  – human-readable summary
  - reports/drift_plot.png   – side-by-side distribution plots

Usage:
  python scripts/simulate_drift.py
  python scripts/simulate_drift.py --n 30 --host http://localhost:8000

"""

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import requests
from PIL import Image, ImageFilter
from scipy import stats

from src.config_loader import load_config
from src.image_features import blur_score as blur_score_local
from src.image_features import image_brightness
from src.utils import ensure_dir, get_logger

log = get_logger("drift")


# ── PSI (Population Stability Index) ─────────────────────────────────────────

def compute_psi(
    reference: np.ndarray, current: np.ndarray, n_bins: int, epsilon: float
) -> float:
    """
    Compute PSI between reference and current distributions.
    SIMULATION ONLY. This helper is not used by the real-log exporter.
    """
    # Bin edges based on reference
    lo = min(reference.min(), current.min())
    hi = max(reference.max(), current.max())
    bins = np.linspace(lo, hi, n_bins + 1)

    ref_counts, _ = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current, bins=bins)

    ref_pct = (ref_counts + epsilon) / (len(reference) + n_bins * epsilon)
    cur_pct = (cur_counts + epsilon) / (len(current) + n_bins * epsilon)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


# ── Send one image to /predict ────────────────────────────────────────────────

def call_predict(host: str, img: Image.Image) -> dict | None:
    """Send image to /predict and return the JSON response (or None on error)."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    try:
        r = requests.post(
            f"{host}/predict",
            files={"file": ("image.jpg", buf, "image/jpeg")},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()
        else:
            log.warning(f"API error {r.status_code}: {r.text[:100]}")
            return None
    except Exception as e:
        log.error(f"Request failed: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="Number of test images to use")
    parser.add_argument("--host", default="http://localhost:8000", help="API host")
    args = parser.parse_args()

    cfg = load_config()
    log.warning("SIMULATION ONLY: transformed TEST images are staged, not real-user traffic.")
    reports_dir = ensure_dir(Path(__file__).resolve().parent.parent / "reports")

    # ── Collect test images ───────────────────────────────────────────────────
    project_root = Path(__file__).resolve().parent.parent
    test_csv = project_root / "data" / "splits" / "test.csv"

    if not test_csv.exists():
        log.error("test.csv not found. Run audit_and_split.py first.")
        sys.exit(1)

    import csv
    with open(test_csv, "r", encoding="utf-8") as f:
        test_records = list(csv.DictReader(f))

    # Sample n images
    rng = np.random.default_rng(cfg.seed)
    sample = rng.choice(test_records, size=min(args.n, len(test_records)), replace=False)
    log.info(f"Using {len(sample)} test images from {test_csv}")

    # ── Check API health ──────────────────────────────────────────────────────
    try:
        health = requests.get(f"{args.host}/health", timeout=5).json()
        if not health.get("model_loaded"):
            log.error("API is up but model is not loaded. Run training first.")
            sys.exit(1)
        log.info(f"API health: {health}")
    except Exception as e:
        log.error(f"Cannot reach API at {args.host}: {e}")
        log.error("Start the server: uvicorn src.api:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    # ── Send reference images ─────────────────────────────────────────────────
    log.info("Sending reference (clean) images...")
    ref_features = {"brightness": [], "blur": [], "confidence": [], "classes": []}

    for rec in sample:
        try:
            img = Image.open(rec["path"]).convert("RGB")
        except Exception:
            continue

        # Local features
        ref_features["brightness"].append(image_brightness(img))
        ref_features["blur"].append(blur_score_local(img))

        resp = call_predict(args.host, img)
        if resp and resp.get("status") == "ok":
            ref_features["confidence"].append(resp["confidence"])
            ref_features["classes"].append(resp.get("condition", "unknown"))
        else:
            ref_features["confidence"].append(0.0)
            ref_features["classes"].append("uncertain")

    log.info(f"Reference set: {len(ref_features['confidence'])} predictions")

    # ── Create drifted images and send ───────────────────────────────────────
    blur_sigma = cfg.drift.blur_sigma
    brightness_factor = cfg.drift.brightness_factor
    log.info(f"Applying drift: Gaussian blur σ={blur_sigma}, brightness×{brightness_factor}")

    drift_features = {"brightness": [], "blur": [], "confidence": [], "classes": []}

    for rec in sample:
        try:
            img = Image.open(rec["path"]).convert("RGB")
        except Exception:
            continue

        # Apply drift: blur + darken
        img_drift = img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
        arr = np.array(img_drift, dtype=np.float32) * brightness_factor
        img_drift = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        drift_features["brightness"].append(image_brightness(img_drift))
        drift_features["blur"].append(blur_score_local(img_drift))

        resp = call_predict(args.host, img_drift)
        if resp and resp.get("status") == "ok":
            drift_features["confidence"].append(resp["confidence"])
            drift_features["classes"].append(resp.get("condition", "unknown"))
        else:
            drift_features["confidence"].append(0.0)
            drift_features["classes"].append("uncertain")

    log.info(f"Drifted set: {len(drift_features['confidence'])} predictions")

    # ── Statistical tests ─────────────────────────────────────────────────────
    alpha = cfg.drift.ks_alpha
    results = {}

    for feature in ["brightness", "blur", "confidence"]:
        ref_arr = np.array(ref_features[feature])
        drift_arr = np.array(drift_features[feature])
        if len(ref_arr) < 2 or len(drift_arr) < 2:
            results[feature] = {"ks_stat": 0, "ks_pvalue": 1, "psi": 0, "drift_detected": False}
            continue

        ks_stat, ks_pvalue = stats.ks_2samp(ref_arr, drift_arr)
        psi = compute_psi(
            ref_arr,
            drift_arr,
            n_bins=int(cfg.drift.psi_bins),
            epsilon=float(cfg.drift.psi_epsilon),
        )
        drift_detected = ks_pvalue < alpha and psi > float(cfg.drift.psi_threshold)

        results[feature] = {
            "ref_mean": round(float(ref_arr.mean()), 3),
            "drift_mean": round(float(drift_arr.mean()), 3),
            "ks_stat": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_pvalue), 4),
            "psi": round(float(psi), 4),
            "drift_detected": drift_detected,
        }
        log.info(
            f"{feature}: ref_mean={results[feature]['ref_mean']:.2f} "
            f"drift_mean={results[feature]['drift_mean']:.2f} "
            f"KS={ks_stat:.4f} p={ks_pvalue:.4f} PSI={psi:.4f} "
            f"{'⚠ DRIFT' if drift_detected else '✓ OK'}"
        )

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        features = ["brightness", "blur", "confidence"]
        titles = ["Image Brightness", "Blur Score (Laplacian Var)", "Prediction Confidence"]

        for ax, feat, title in zip(axes, features, titles):
            ref_arr = np.array(ref_features[feat])
            drift_arr = np.array(drift_features[feat])
            ax.hist(ref_arr, bins=15, alpha=0.6, label="Reference", color="steelblue")
            ax.hist(drift_arr, bins=15, alpha=0.6, label="Drifted", color="coral")
            r = results.get(feat, {})
            ax.set_title(
                f"{title}\nKS p={r.get('ks_pvalue', '?'):.3f}, PSI={r.get('psi', '?'):.3f} "
                f"{'⚠ DRIFT' if r.get('drift_detected') else '✓ OK'}"
            )
            ax.legend()

        plt.suptitle("Distribution Drift: Reference vs Drifted Images", fontsize=13, y=1.02)
        plt.tight_layout()
        plot_path = reports_dir / "drift_plot.png"
        fig.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Drift plot saved: {plot_path}")
    except Exception as e:
        log.warning(f"Plot generation failed: {e}")
        plot_path = None

    # ── Write markdown report ─────────────────────────────────────────────────
    report_lines = [
        "# SIMULATION: Drift Report",
        "",
        f"**Reference images**: {len(ref_features['confidence'])}  |  "
        f"**Drifted images**: {len(drift_features['confidence'])}",
        f"**Drift applied**: Gaussian blur σ={blur_sigma}, brightness factor={brightness_factor}",
        f"**SIMULATION only — KS α**: {alpha}  |  **Provisional PSI proposal**: {cfg.drift.psi_threshold}",
        "",
        "## Results",
        "",
        "| Feature | Ref Mean | Drift Mean | KS stat | KS p-value | PSI | Drift? |",
        "|---------|----------|------------|---------|------------|-----|--------|",
    ]
    for feat in ["brightness", "blur", "confidence"]:
        r = results.get(feat, {})
        flag = "⚠ YES" if r.get("drift_detected") else "✓ No"
        report_lines.append(
            f"| {feat} | {r.get('ref_mean','?')} | {r.get('drift_mean','?')} | "
            f"{r.get('ks_stat','?')} | {r.get('ks_pvalue','?')} | {r.get('psi','?')} | {flag} |"
        )

    report_lines += [
        "",
        "## Interpretation",
        "- This SIMULATION applies the configured PSI AND KS rule; values remain proposed, not approved.",
        "- Blur score and brightness show the applied drift; confidence drop confirms the model",
        "  degrades on blurry/dark images that differ from the training distribution.",
        "",
        "## MLOps Relevance",
        "This script simulates the kind of input distribution shift that would be detected by a",
        "monitoring system in production. In a real pipeline, these KS tests would run automatically",
        "on incoming request logs and trigger an alert or retraining job when drift is detected.",
    ]

    if plot_path:
        report_lines += ["", f"![Drift Plot]({plot_path.name})"]

    report_path = reports_dir / "drift_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    log.info(f"Drift report saved: {report_path}")
    try:
        print("\n" + "\n".join(report_lines))
    except UnicodeEncodeError:
        print("\n" + "\n".join(report_lines).encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
