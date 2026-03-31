#!/usr/bin/env python
"""
Step 1 — Feature Extraction
============================
Extract PyRadiomics features from CT NIfTI images and their segmentation masks.
Run this once for train data (with --labels-csv) and once for test data (without).

Usage — train data
------------------
    python extract_features.py \\
        --config config.yaml \\
        --images /path/to/imagesTr \\
        --masks  /path/to/labelsTr \\
        --labels-csv /path/to/labels.csv \\
        --output-csv features/train_features.csv

Usage — test data  (no --labels-csv; no 'class' column written)
---------------------------------------------------------------
    python extract_features.py \\
        --config config.yaml \\
        --images /path/to/imagesTs \\
        --masks  /path/to/labelsTs \\
        --output-csv features/test_features.csv

Notes
-----
- Images and masks are matched by patient_id (filename without .nii / .nii.gz).
- The mask label used as the tumour ROI is set by config['data']['label_value']
  (default 2 = tumour in Task 1 output masks).
- Override label_value with --label-value if needed.
- n_jobs in config['feature_extraction'] controls parallelism.
  Set n_jobs: 1 on Windows to avoid multiprocessing issues with pyradiomics.
"""
import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

# Ensure the radiomics_baseline package root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))
from src.feature_extractor import RadiomicsExtractor


# ---------------------------------------------------------------------------
# Worker (must be at module level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------

def _extract_one(args):
    """Extract features for a single patient (picklable top-level function)."""
    patient_id, image_path, mask_path, config = args
    extractor = RadiomicsExtractor(config)
    features  = extractor.extract(image_path, mask_path)
    return patient_id, features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_nii_map(directory):
    """Return {patient_id: full_path} for every .nii / .nii.gz in *directory*."""
    mapping = {}
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".nii.gz"):
            pid = fname[:-7]
        elif fname.endswith(".nii"):
            pid = fname[:-4]
        else:
            continue
        mapping[pid] = os.path.join(directory, fname)
    return mapping


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract PyRadiomics features from NIfTI CT + mask pairs."
    )
    parser.add_argument("--config",       default="config.yaml",
                        help="Path to config.yaml  (default: config.yaml)")
    parser.add_argument("--images",       required=True,
                        help="Directory with <patient_id>.nii.gz CT images")
    parser.add_argument("--masks",        required=True,
                        help="Directory with <patient_id>.nii.gz segmentation masks")
    parser.add_argument("--labels-csv",   default=None,
                        help="CSV with patient_id + type/class column (train only)")
    parser.add_argument("--output-csv",   required=True,
                        help="Output feature CSV path")
    parser.add_argument("--label-value",  type=int, default=None,
                        help="Tumour label integer in mask (overrides config)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    # Load config
    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    if args.label_value is not None:
        config["data"]["label_value"] = args.label_value

    # Build patient-id → file-path maps and find the intersection
    image_map = _build_nii_map(args.images)
    mask_map  = _build_nii_map(args.masks)
    patient_ids = sorted(set(image_map) & set(mask_map))

    if not patient_ids:
        logging.error(
            "No matching image/mask pairs found.  "
            "Check that --images and --masks point to directories with "
            "matching .nii.gz filenames."
        )
        sys.exit(1)

    logging.info("Found %d image/mask pairs.", len(patient_ids))

    # Optional label map (for train data)
    labels: dict = {}
    if args.labels_csv:
        ldf       = pd.read_csv(args.labels_csv)
        label_col = "type" if "type" in ldf.columns else "class"
        labels    = dict(zip(ldf["patient_id"].astype(str), ldf[label_col]))
        logging.info("Loaded %d labels from %s.", len(labels), args.labels_csv)

    # Feature extraction
    n_jobs = config.get("feature_extraction", {}).get("n_jobs", 1)
    tasks  = [
        (pid, image_map[pid], mask_map[pid], config)
        for pid in patient_ids
    ]

    rows:   list = []
    failed: list = []

    if n_jobs not in (0, 1):
        # Parallel extraction via ProcessPoolExecutor
        workers = n_jobs if n_jobs > 0 else None
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_extract_one, t): t[0] for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="Extracting", unit="patient"):
                pid = futures[fut]
                try:
                    pid, features = fut.result()
                except Exception as exc:
                    logging.error("%s: %s", pid, exc)
                    failed.append(pid)
                    continue

                if features is None:
                    failed.append(pid)
                    continue

                row = {"patient_id": pid}
                if pid in labels:
                    row["class"] = labels[pid]
                row.update(features)
                rows.append(row)
    else:
        # Sequential extraction (safer on Windows / debugging)
        extractor = RadiomicsExtractor(config)
        for pid, image_path, mask_path, _ in tqdm(tasks,
                                                   desc="Extracting",
                                                   unit="patient"):
            features = extractor.extract(image_path, mask_path)

            if features is None:
                failed.append(pid)
                continue

            row = {"patient_id": pid}
            if pid in labels:
                row["class"] = labels[pid]
            row.update(features)
            rows.append(row)

    if not rows:
        logging.error(
            "No features were successfully extracted.  "
            "Check label_value (currently %d) and mask files.",
            config["data"]["label_value"],
        )
        sys.exit(1)

    df = pd.DataFrame(rows)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    n_feat = len(df.columns) - (2 if "class" in df.columns else 1)
    logging.info(
        "Saved %d patients x %d features → %s", len(df), n_feat, out_path
    )
    if failed:
        logging.warning(
            "%d patient(s) failed extraction: %s", len(failed), failed
        )


if __name__ == "__main__":
    main()
