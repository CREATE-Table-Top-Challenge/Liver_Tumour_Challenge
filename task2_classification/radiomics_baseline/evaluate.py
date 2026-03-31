#!/usr/bin/env python
"""
Step 3 — Evaluation / Submission
==================================
Load a trained pipeline (or k-fold ensemble) and run inference on the
pre-extracted test features.  Saves a submission CSV in exactly the same
format as task2_classification/evaluate.py.

The user must extract test features first:
    python extract_features.py --config config.yaml \\
        --images /path/imagesTs --masks /path/labelsTs \\
        --output-csv features/test_features.csv

Then run this script:

Usage — after k-fold training  (recommended)
----------------------------------------------------------------------
    python evaluate.py \\
        --config config.yaml \\
        --features features/test_features.csv \\
        --models-dir model_checkpoints/radiomics_rf_baseline

    When cv_results.json is present, the final model (trained on all data)
    is used automatically.  The fold ensemble is a fallback if the final
    model file is missing.

Usage — single model
---------------------
    python evaluate.py \\
        --config config.yaml \\
        --features features/test_features.csv \\
        --model model_checkpoints/radiomics_rf_baseline/pipeline.joblib

Output
------
    results/group<N>_task2_results.csv
    Columns: patient_id, predicted_class, prob_BCLM, prob_CRLM, prob_HCC,
             prob_HH, prob_ICC
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_test_features(features_csv):
    """
    Load the test feature CSV.  No 'class' column is expected.

    Returns
    -------
    X           : float32 ndarray  (n_samples, n_features)
    patient_ids : ndarray of str
    """
    df = pd.read_csv(features_csv)
    patient_ids  = df["patient_id"].astype(str).values
    feature_cols = [c for c in df.columns if c not in ("patient_id", "class")]
    X = df[feature_cols].fillna(0.0).values.astype(np.float32)
    return X, patient_ids


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _predict_single(pipeline, X):
    return pipeline.predict(X), pipeline.predict_proba(X)


def _predict_ensemble(pipelines, X):
    """Soft-vote: average predict_proba across all fold pipelines."""
    probs      = np.stack([p.predict_proba(X) for p in pipelines], axis=0)
    mean_probs = probs.mean(axis=0)
    return mean_probs.argmax(axis=1), mean_probs


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _save_submission_csv(patient_ids, y_pred, y_prob,
                          class_names, output_dir, group_number):
    """
    Write group<N>_task2_results.csv in the organiser-expected format.

    Columns: patient_id, predicted_class, prob_BCLM, …, prob_ICC
    """
    idx_to_class = {i: c for i, c in enumerate(class_names)}
    rows = []
    for pid, pred_idx, probs in zip(patient_ids, y_pred, y_prob):
        row = {
            "patient_id":      pid,
            "predicted_class": idx_to_class[int(pred_idx)],
        }
        for i, cname in enumerate(class_names):
            row[f"prob_{cname}"] = f"{probs[i]:.6f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    csv_name = f"group{group_number}_task2_results.csv"
    out_path = os.path.join(output_dir, csv_name)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} predictions → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Radiomics baseline — inference and submission CSV generation"
    )
    p.add_argument("--config",      default="config.yaml",
                   help="Path to config.yaml  (default: config.yaml)")
    p.add_argument("--features",    required=True,
                   help="Test feature CSV produced by extract_features.py")
    p.add_argument("--model",       default=None,
                   help="Path to a single pipeline.joblib")
    p.add_argument("--models-dir",  default=None,
                   help="Directory containing cv_results.json (k-fold ensemble)")
    p.add_argument("--output-dir",  default=None,
                   help="Override results directory  (default: from config)")
    return p.parse_args()


def main():
    args = _parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    class_names  = config["data"]["class_names"]
    group_number = config.get("group_number", 0)
    output_dir   = args.output_dir or config["output"]["results_dir"]

    # Load test features
    X, patient_ids = _load_test_features(args.features)
    print(f"Loaded {len(patient_ids)} test samples with {X.shape[1]} features.")

    # -----------------------------------------------------------------
    # Model loading — three fallback modes:
    #   1. --models-dir with cv_results.json  →  k-fold soft-vote ensemble
    #   2. --model explicit path              →  single pipeline
    #   3. pipeline.joblib inside --models-dir →  single pipeline (fallback)
    # -----------------------------------------------------------------
    cv_json = (
        os.path.join(args.models_dir, "cv_results.json")
        if args.models_dir else None
    )

    if cv_json and os.path.exists(cv_json):
        with open(cv_json) as fh:
            cv_results = json.load(fh)

        # Prefer the final model (trained on all data) over the fold ensemble
        final_path = cv_results.get("final_pipeline_path")
        if final_path and os.path.exists(final_path):
            pipeline = joblib.load(final_path)
            print(f"Loaded final model (trained on all data) from {final_path}")
            y_pred, y_prob = _predict_single(pipeline, X)
        else:
            # Fall back to soft-vote ensemble across folds
            pipelines = []
            for fold_path in cv_results["fold_pipeline_paths"]:
                if os.path.exists(fold_path):
                    pipelines.append(joblib.load(fold_path))
                else:
                    print(f"  Warning: fold model not found at {fold_path} — skipped.")

            if not pipelines:
                raise FileNotFoundError(
                    "No fold models found via cv_results.json.  "
                    "Run train.py first."
                )
            print(f"Loaded {len(pipelines)}-fold ensemble from {args.models_dir}")
            y_pred, y_prob = _predict_ensemble(pipelines, X)

    elif args.model and os.path.exists(args.model):
        pipeline = joblib.load(args.model)
        print(f"Loaded single model from {args.model}")
        y_pred, y_prob = _predict_single(pipeline, X)

    else:
        # Last-resort: look for pipeline.joblib inside --models-dir
        fallback = (
            os.path.join(args.models_dir, "pipeline.joblib")
            if args.models_dir else None
        )
        if fallback and os.path.exists(fallback):
            pipeline = joblib.load(fallback)
            print(f"Loaded single model from {fallback}")
            y_pred, y_prob = _predict_single(pipeline, X)
        else:
            raise ValueError(
                "No model found.  "
                "Provide --model <path> or --models-dir <dir-with-cv_results.json>."
            )

    # Quick per-class breakdown
    idx_to_class  = {i: c for i, c in enumerate(class_names)}
    pred_classes  = [idx_to_class[int(p)] for p in y_pred]
    counts        = Counter(pred_classes)
    print("\nPrediction distribution:")
    for cls in class_names:
        print(f"  {cls:<6}: {counts.get(cls, 0)}")

    _save_submission_csv(
        patient_ids, y_pred, y_prob,
        class_names, output_dir, group_number,
    )


if __name__ == "__main__":
    main()
