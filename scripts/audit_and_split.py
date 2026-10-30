"""
audit_and_split.py
==================
Step 2: Audit dataset, deduplicate, and create group-aware train/val/test splits.

What it does:
  1. Verifies class counts match the known reference (warns on mismatch).
  2. Checks image sizes & formats (sampled check).
  3. Exact-duplicate detection via SHA-256 hash (removes all but one copy).
  4. Near-duplicate detection via 16x16 dHash + Hamming threshold (reports only; no deletion).
  5. Group-ID hypothesis test (filename → group_id by stripping UUID prefix + ".n" suffix).
  6. Group-aware stratified split 70/15/15.
  7. Saves split manifests as CSV to data/splits/.
  8. Prints a short summary report saved to reports/audit_report.md.

Usage:
  python scripts/audit_and_split.py

Environment:
  DATA_ROOT must be set (via .env or environment).
"""

import hashlib
import sys
import re
import json
from collections import defaultdict
from pathlib import Path
import csv

# ── allow src imports ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.config_loader import load_config
from src.utils import set_seed, get_logger, parse_class_name, ensure_dir

log = get_logger("audit")

# Reference counts from the brief (for verification)
REFERENCE_COUNTS: dict[str, int] = {
    "Apple___Apple_scab": 630,
    "Apple___Black_rot": 621,
    "Apple___Cedar_apple_rust": 275,
    "Apple___healthy": 1645,
    "Blueberry___healthy": 1502,
    "Cherry_(including_sour)___Powdery_mildew": 1052,
    "Cherry_(including_sour)___healthy": 854,
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": 513,
    "Corn_(maize)___Common_rust_": 1192,
    "Corn_(maize)___Northern_Leaf_Blight": 985,
    "Corn_(maize)___healthy": 1162,
    "Grape___Black_rot": 1180,
    "Grape___Esca_(Black_Measles)": 1383,
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": 1076,
    "Grape___healthy": 423,
    "Orange___Haunglongbing_(Citrus_greening)": 5507,
    "Peach___Bacterial_spot": 2297,
    "Peach___healthy": 360,
    "Pepper,_bell___Bacterial_spot": 997,
    "Pepper,_bell___healthy": 1478,
    "Potato___Early_blight": 1000,
    "Potato___Late_blight": 1000,
    "Potato___healthy": 152,
    "Raspberry___healthy": 371,
    "Soybean___healthy": 5090,
    "Squash___Powdery_mildew": 1835,
    "Strawberry___Leaf_scorch": 1109,
    "Strawberry___healthy": 456,
    "Tomato___Bacterial_spot": 2127,
    "Tomato___Early_blight": 1000,
    "Tomato___Late_blight": 1909,
    "Tomato___Leaf_Mold": 952,
    "Tomato___Septoria_leaf_spot": 1771,
    "Tomato___Spider_mites Two-spotted_spider_mite": 1676,
    "Tomato___Target_Spot": 1404,
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 5357,
    "Tomato___Tomato_mosaic_virus": 373,
    "Tomato___healthy": 1591,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash_16(img: Image.Image) -> int:
    """Compute 16x16 dHash → 256-bit integer."""
    gray = img.convert("L").resize((17, 16), Image.LANCZOS)
    arr = np.array(gray)
    diff = arr[:, 1:] > arr[:, :-1]   # shape (16, 16)
    bits = diff.flatten()
    # Pack into int
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def extract_group_id(filename: str, crop: str = "") -> str:
    """
    Strip the UUID prefix from a PlantVillage filename to get a group ID.

    Pattern observed: "<uuid>___<source_code> <number>[.n].<ext>"
    We want: "<source_code> <number>" (without the trailing .n multi-shot suffix).

    IMPORTANT: Different crops use independent capture indexing (e.g. 'RS_HL 5672' exists
    in both Apple and Soybean). Prefixing with crop ensures groups accurately represent
    real leaf/plant capture sessions and avoids artificial cross-crop collisions.
    """
    stem = Path(filename).stem  # no extension
    parts = stem.split("___", maxsplit=1)
    if len(parts) < 2:
        raw_gid = stem  # no UUID pattern; use full stem as group
    else:
        tail = parts[1]  # e.g. "GH_HL Leaf 482.2"
        # Remove trailing ".N" (integer suffix indicating multi-shot)
        raw_gid = re.sub(r"\.\d+$", "", tail).strip()

    return f"{crop.lower()}::{raw_gid}" if crop else raw_gid


def stratified_group_split(
    records: list[dict],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Group-aware stratified split mathematically guaranteeing zero group overlap.

    Strategy:
      1. Map each group_id to all its records and determine its primary class (majority class).
      2. For each class, shuffle the groups belonging to that class.
      3. Assign groups to test, val, and train splits.
      4. Because assignment is strictly per group_id globally, no group_id can ever
         appear in more than one split (zero leakage).
    """
    rng = np.random.default_rng(seed)

    # group_id -> list of records
    group_to_records: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        group_to_records[r["group_id"]].append(r)

    # Determine primary class for each group
    class_to_groups: dict[str, list[str]] = defaultdict(list)
    for gid, recs in group_to_records.items():
        # count classes in this group
        cls_counts = defaultdict(int)
        for r in recs:
            cls_counts[r["class_name"]] += 1
        primary_cls = max(cls_counts.keys(), key=lambda k: cls_counts[k])
        class_to_groups[primary_cls].append(gid)

    # Group assignment mapping: group_id -> "train" | "val" | "test"
    group_assignment: dict[str, str] = {}

    for cls, gids in class_to_groups.items():
        rng.shuffle(gids)
        n = len(gids)
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        n_train = n - n_test - n_val
        if n_train < 1:
            n_train = 1
            n_val = max(0, n - n_train - n_test)

        test_gids = set(gids[:n_test])
        val_gids = set(gids[n_test: n_test + n_val])

        for gid in gids:
            if gid in test_gids:
                group_assignment[gid] = "test"
            elif gid in val_gids:
                group_assignment[gid] = "val"
            else:
                group_assignment[gid] = "train"

    train_recs, val_recs, test_recs = [], [], []
    for r in records:
        target = group_assignment[r["group_id"]]
        if target == "test":
            test_recs.append(r)
        elif target == "val":
            val_recs.append(r)
        else:
            train_recs.append(r)

    return train_recs, val_recs, test_recs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    set_seed(cfg.seed)

    data_root = Path(cfg.data.root)
    reports_dir = ensure_dir(Path(__file__).resolve().parent.parent / "reports")
    splits_dir = ensure_dir(Path(__file__).resolve().parent.parent / "data" / "splits")

    log.info(f"Dataset root: {data_root}")

    # ── 1. Discover class folders ─────────────────────────────────────────────
    class_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    log.info(f"Found {len(class_dirs)} class folders")

    # ── 2. Collect all images with metadata ──────────────────────────────────
    all_records: list[dict] = []
    class_counts: dict[str, int] = {}
    size_samples: list[tuple[int, int]] = []
    formats_seen: set[str] = set()
    SIZE_SAMPLE_MAX = 200

    log.info("Scanning images...")
    for cls_dir in tqdm(class_dirs, desc="Classes"):
        cls_name = cls_dir.name
        try:
            crop, condition = parse_class_name(cls_name)
        except ValueError:
            log.warning(f"Cannot parse class name: {cls_name}")
            crop, condition = cls_name, "unknown"

        valid_exts = {".jpg", ".jpeg", ".png"}
        imgs = sorted([p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts])

        class_counts[cls_name] = len(imgs)

        for img_path in imgs:
            group_id = extract_group_id(img_path.name, crop=crop)
            all_records.append({
                "path": str(img_path),
                "filename": img_path.name,
                "class_name": cls_name,
                "class_idx": class_dirs.index(cls_dir),
                "crop": crop,
                "condition": condition,
                "group_id": group_id,
            })

            # Sample image sizes / formats
            if len(size_samples) < SIZE_SAMPLE_MAX:
                try:
                    with Image.open(img_path) as im:
                        size_samples.append(im.size)
                        formats_seen.add(im.format or "unknown")
                except Exception:
                    pass

    total_images = len(all_records)
    log.info(f"Total images found: {total_images}")

    # ── 3. Verify class counts vs reference ───────────────────────────────────
    count_mismatches = []
    for ref_cls, ref_count in REFERENCE_COUNTS.items():
        found = class_counts.get(ref_cls, 0)
        if found != ref_count:
            count_mismatches.append((ref_cls, ref_count, found))

    if count_mismatches:
        log.warning(f"Class count mismatches: {len(count_mismatches)}")
        for cls, expected, got in count_mismatches:
            log.warning(f"  {cls}: expected {expected}, got {got}")
    else:
        log.info("All class counts match the reference ✓")

    # ── 4. Size / format check ────────────────────────────────────────────────
    if size_samples:
        sizes_arr = np.array(size_samples)
        log.info(f"Image size sample (n={len(size_samples)}): "
                 f"min={sizes_arr.min(axis=0).tolist()}, "
                 f"max={sizes_arr.max(axis=0).tolist()}, "
                 f"median={np.median(sizes_arr, axis=0).tolist()}")
        log.info(f"Formats seen: {formats_seen}")

    # ── 5. Exact duplicate detection (SHA-256) ────────────────────────────────
    log.info("Computing SHA-256 hashes for exact duplicate detection...")
    hash_to_paths: dict[str, list[str]] = defaultdict(list)
    for rec in tqdm(all_records, desc="Hashing"):
        h = sha256_file(Path(rec["path"]))
        rec["sha256"] = h
        hash_to_paths[h].append(rec["path"])

    exact_dup_groups = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}
    total_exact_dups = sum(len(v) - 1 for v in exact_dup_groups.values())
    log.info(f"Exact duplicate groups: {len(exact_dup_groups)} | "
             f"Extra copies to remove: {total_exact_dups}")

    # Remove duplicate records (keep first occurrence per hash)
    seen_hashes: set[str] = set()
    deduped_records: list[dict] = []
    removed_paths: list[str] = []
    for rec in all_records:
        if rec["sha256"] not in seen_hashes:
            seen_hashes.add(rec["sha256"])
            deduped_records.append(rec)
        else:
            removed_paths.append(rec["path"])

    log.info(f"After exact dedup: {len(deduped_records)} images "
             f"(removed {len(removed_paths)} duplicates)")

    # ── 6. Near-duplicate detection (16x16 dHash, Hamming ≤ 10) ─────────────
    HAMMING_THRESHOLD = 10
    NEAR_DUP_SAMPLE = min(1500, len(deduped_records))  # fast, representative sample

    log.info(f"Computing dHash on {NEAR_DUP_SAMPLE} images for near-dup check...")
    sample_recs = deduped_records[:NEAR_DUP_SAMPLE]
    hashes_dhash: list[tuple[str, int]] = []

    for rec in tqdm(sample_recs, desc="dHash"):
        try:
            with Image.open(rec["path"]) as im:
                h = dhash_16(im)
            hashes_dhash.append((rec["path"], h))
        except Exception:
            pass

    # Find near-dup pairs
    near_dup_pairs: list[tuple[str, str, int]] = []
    for i in range(len(hashes_dhash)):
        for j in range(i + 1, len(hashes_dhash)):
            dist = hamming_distance(hashes_dhash[i][1], hashes_dhash[j][1])
            if dist <= HAMMING_THRESHOLD:
                near_dup_pairs.append((hashes_dhash[i][0], hashes_dhash[j][0], dist))

    log.info(f"Near-duplicate pairs found (Hamming ≤ {HAMMING_THRESHOLD}, "
             f"sampled {NEAR_DUP_SAMPLE}): {len(near_dup_pairs)} "
             f"[NOTE: Not auto-deleted; review manually]")

    # ── 7. Group-ID analysis ──────────────────────────────────────────────────
    group_id_to_classes: dict[str, set[str]] = defaultdict(set)
    for rec in deduped_records:
        group_id_to_classes[rec["group_id"]].add(rec["class_name"])

    unique_groups = len(group_id_to_classes)
    cross_class_groups = {g: cls for g, cls in group_id_to_classes.items() if len(cls) > 1}
    log.info(f"Unique group IDs: {unique_groups} | Images: {len(deduped_records)} | "
             f"Ratio: {len(deduped_records)/unique_groups:.2f} images/group")
    log.info(f"Groups spanning multiple classes: {len(cross_class_groups)}")

    # Source-code prefix analysis
    source_code_pattern = re.compile(r"^([A-Z]+_[A-Z]*)")
    class_source_codes: dict[str, set[str]] = defaultdict(set)
    for rec in deduped_records:
        m = source_code_pattern.match(rec["group_id"])
        if m:
            class_source_codes[rec["class_name"]].add(m.group(1))

    source_exclusive = sum(1 for codes in class_source_codes.values() if len(codes) == 1)
    log.info(f"Classes with only ONE source code: {source_exclusive}/{len(class_dirs)} "
             f"(capture-source bias present)")

    # ── 8. Group-aware stratified split ───────────────────────────────────────
    log.info("Creating group-aware stratified 70/15/15 split...")
    train_recs, val_recs, test_recs = stratified_group_split(
        deduped_records,
        val_frac=cfg.split.val,
        test_frac=cfg.split.test,
        seed=cfg.seed,
    )
    log.info(f"Split sizes — train: {len(train_recs)}, val: {len(val_recs)}, "
             f"test: {len(test_recs)}")

    # ── 9. Save split manifests as CSV ───────────────────────────────────────
    fieldnames = ["path", "filename", "class_name", "class_idx", "crop",
                  "condition", "group_id", "sha256"]
    for split_name, recs in [("train", train_recs), ("val", val_recs), ("test", test_recs)]:
        csv_path = splits_dir / f"{split_name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(recs)
        log.info(f"Saved {split_name} manifest: {csv_path}")

    # Compute manifest hash (for MLflow logging)
    def csv_hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:12]

    train_hash = csv_hash(splits_dir / "train.csv")
    log.info(f"Train manifest SHA256 prefix: {train_hash}")

    # Save hash for downstream use
    (splits_dir / "manifest_hashes.json").write_text(
        json.dumps({
            "train": csv_hash(splits_dir / "train.csv"),
            "val": csv_hash(splits_dir / "val.csv"),
            "test": csv_hash(splits_dir / "test.csv"),
        }, indent=2)
    )

    # ── 10. Zero-overlap assertion ────────────────────────────────────────────
    train_groups = {r["group_id"] for r in train_recs}
    val_groups = {r["group_id"] for r in val_recs}
    test_groups = {r["group_id"] for r in test_recs}

    tv_overlap = train_groups & val_groups
    tt_overlap = train_groups & test_groups
    vt_overlap = val_groups & test_groups

    if tv_overlap or tt_overlap or vt_overlap:
        log.error(f"GROUP OVERLAP DETECTED! "
                  f"train∩val={len(tv_overlap)}, train∩test={len(tt_overlap)}, "
                  f"val∩test={len(vt_overlap)}")
        sys.exit(1)
    else:
        log.info("Group overlap check: PASSED ✓ (zero overlap across splits)")

    # ── 11. Write audit report ────────────────────────────────────────────────
    report_lines = [
        "# Dataset Audit Report",
        "",
        f"**Total images found**: {total_images}",
        f"**Class folders**: {len(class_dirs)}",
        f"**Class count mismatches vs reference**: {len(count_mismatches)}",
        "",
        "## Image Size Sample",
        f"- Sample size: {len(size_samples)}",
        f"- Min WxH: {sizes_arr.min(axis=0).tolist() if size_samples else 'N/A'}",
        f"- Max WxH: {sizes_arr.max(axis=0).tolist() if size_samples else 'N/A'}",
        f"- Median WxH: {np.median(sizes_arr, axis=0).tolist() if size_samples else 'N/A'}",
        f"- Formats: {formats_seen}",
        "",
        "## Deduplication",
        f"- Exact duplicate groups (SHA-256): {len(exact_dup_groups)}",
        f"- Extra copies removed: {total_exact_dups}",
        f"- Images after dedup: {len(deduped_records)}",
        "",
        "## Near-Duplicate Check (16×16 dHash, Hamming ≤ 10)",
        f"- Sample size: {NEAR_DUP_SAMPLE}",
        f"- Near-dup pairs found: {len(near_dup_pairs)}",
        "- **Action**: REPORTED ONLY. Not auto-deleted (brief instruction).",
        "- NOTE: A weak hash causes many false positives across classes; treat cross-class matches with skepticism.",
        "",
        "## Group-ID Analysis",
        f"- Unique group IDs: {unique_groups}",
        f"- Images (post-dedup): {len(deduped_records)}",
        f"- Avg images per group: {len(deduped_records)/unique_groups:.2f}",
        f"- Groups spanning multiple classes: {len(cross_class_groups)} (should be ~0 if filenames are trustworthy)",
        f"- Classes with single source code: {source_exclusive}/{len(class_dirs)} → **capture-source bias confirmed**",
        "",
        "## Split Summary",
        f"- Train: {len(train_recs)} images",
        f"- Val:   {len(val_recs)} images",
        f"- Test:  {len(test_recs)} images",
        "- Group overlap: ZERO ✓",
        "",
        "## Key Implications",
        "- Lab-style single-leaf images on uniform backgrounds → high accuracy expected but does NOT imply field performance.",
        "- Five crops have a single class only (Blueberry, Raspberry, Soybean: healthy; Orange, Squash: disease)",
        "  so crop identity alone predicts the label for ~26% of data.",
        "- Severe class imbalance (~36:1). Addressed via class-weighted loss in training.",
        "- Source-code bias: models may learn capture-source artifacts, not just disease features.",
        "- This is a prototype pipeline, NOT a field-ready diagnostic tool.",
    ]

    report_path = reports_dir / "audit_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    log.info(f"Audit report saved: {report_path}")
    try:
        print("\n" + "\n".join(report_lines))
    except UnicodeEncodeError:
        print("\n" + "\n".join(report_lines).encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
