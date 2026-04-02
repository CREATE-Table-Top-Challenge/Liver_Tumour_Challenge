#!/usr/bin/env python
"""
Step 1 — Feature Extraction from ROIs
==================================================
Extract PyRadiomics features from tumor ROI NIfTI files
(ROIs are already masked and cropped).

Usage — train data (with class labels)
--------------------------------------
    python extract_features.py \\
        --config config.yaml \\
        --rois /path/to/train/roi_data \\
        --labels-csv /path/to/labels.csv \\
        --output-csv features/train_features.csv

Usage — test data (no labels)
-----------------------------
    python extract_features.py \\
        --config config.yaml \\
        --rois /path/to/test/roi_data \\
        --output-csv features/test_features.csv

Notes
-----
- ROI NIfTI files contain masked tumor regions.
- For train data, --labels-csv is required and adds a 'class' column.
- For test data, no labels are needed; output contains features only.
- n_jobs in config['feature_extraction'] controls parallelism.
  Set n_jobs: 1 on Windows to avoid multiprocessing issues.
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
    """Extract features for a single ROI (picklable top-level function)."""
    patient_id, roi_path, config = args
    extractor = RadiomicsExtractor(config)
    features  = extractor.extract_from_roi(roi_path)
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
        description="Extract PyRadiomics features from ROI NIfTI files."
    )
    parser.add_argument("--config",       default="config.yaml",
                        help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--rois",         required=True,
                        help="Directory with ROI NIfTI files (*.nii.gz or *.nii)")
    parser.add_argument("--labels-csv",   default=None,
                        help="CSV with patient_id + type/class column (train only)")
    parser.add_argument("--output-csv",   required=True,
                        help="Output feature CSV path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    # Load config
    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    # Build ROI file map
    roi_map = _build_nii_map(args.rois)
    patient_ids = sorted(roi_map.keys())

    if not patient_ids:
        logging.error(
            "No ROI NIfTI files found in %s. "
            "Check that the directory contains .nii.gz or .nii files.",
            args.rois
        )
        sys.exit(1)

    logging.info("Found %d ROI files.", len(patient_ids))

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
        (pid, roi_map[pid], config)
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
