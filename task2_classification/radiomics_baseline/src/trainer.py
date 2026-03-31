"""
Training logic for the radiomics baseline.

Public API
----------
train_single(features_csv, config, output_dir)
    Train on a single train/val split.

train_kfold(features_csv, config, output_dir)
    Train with N-fold cross-validation.
    Each fold saves to output_dir/fold_N/pipeline.joblib.
    A cv_results.json manifest is written to output_dir/ on completion.
    If a fold's fold_complete.json marker already exists the fold is skipped,
    making it safe to resume an interrupted run.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

from .classifier import build_pipeline
from .metrics import compute_metrics, print_metrics


# ---------------------------------------------------------------------------
# Fine-tuning helper
# ---------------------------------------------------------------------------

def _maybe_finetune(pipeline, X_train, y_train, config):
    """
    If fine_tuning.enabled is true in config, wrap *pipeline* in a
    GridSearchCV and fit it; otherwise just fit the pipeline directly.

    Returns the fitted estimator (either the pipeline or the GridSearchCV).
    """
    ft_cfg = config.get("fine_tuning", {})
    if not ft_cfg.get("enabled", False):
        pipeline.fit(X_train, y_train)
        return pipeline

    param_grid = ft_cfg.get("param_grid", {})
    if not param_grid:
        raise ValueError(
            "fine_tuning.enabled is true but fine_tuning.param_grid is empty."
        )

    gs = GridSearchCV(
        pipeline,
        param_grid  = param_grid,
        cv          = ft_cfg.get("cv", 3),
        scoring     = ft_cfg.get("scoring", "f1_macro"),
        refit       = True,
        n_jobs      = -1,
        verbose     = 1,
    )
    gs.fit(X_train, y_train)
    print(f"  Best params  : {gs.best_params_}")
    print(f"  Best CV score: {gs.best_score_:.4f}")
    return gs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_features(features_csv, class_names):
    """
    Load a feature CSV that must contain 'patient_id' and 'class' columns.

    Returns
    -------
    X            : float32 ndarray  (n_samples, n_features)
    y            : int ndarray      (n_samples,)
    patient_ids  : ndarray of str
    feature_cols : list[str]
    """
    df = pd.read_csv(features_csv)

    if "class" not in df.columns:
        raise ValueError(
            f"'class' column not found in {features_csv}. "
            "Make sure you ran extract_features.py with --labels-csv."
        )

    class_to_idx = {c: i for i, c in enumerate(class_names)}
    patient_ids  = df["patient_id"].astype(str).values
    y            = df["class"].map(class_to_idx).values

    unknown = set(df["class"].unique()) - set(class_names)
    if unknown:
        raise ValueError(f"Unknown class labels in CSV: {unknown}")

    feature_cols = [c for c in df.columns if c not in ("patient_id", "class")]
    X = df[feature_cols].fillna(0.0).values.astype(np.float32)

    return X, y, patient_ids, feature_cols


def _save_artefacts(estimator, feature_cols, metrics, output_dir, config):
    """
    Persist the fitted estimator and companion artefacts to *output_dir*.

    Artefacts
    ---------
    pipeline.joblib          — fitted pipeline (or GridSearchCV wrapping it)
    results.json             — val metrics
    feature_importance.csv   — per-feature importance (RandomForest / XGBoost)
    """
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(estimator, os.path.join(output_dir, "pipeline.joblib"))

    # Unwrap GridSearchCV to reach the underlying pipeline's classifier
    pipeline = getattr(estimator, "best_estimator_", estimator)
    clf = pipeline.named_steps["classifier"]
    if hasattr(clf, "feature_importances_"):
        fi = clf.feature_importances_
        fi_df = pd.DataFrame({
            "feature":    feature_cols[: len(fi)],
            "importance": fi,
        }).sort_values("importance", ascending=False)
        fi_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)

    result_dict = {
        "classifier":   config["classifier"]["type"],
        "val_accuracy": metrics["accuracy"],
        "val_f1_macro": metrics["f1_macro"],
        "metrics":      metrics,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as fh:
        json.dump(result_dict, fh, indent=2)

    return result_dict


# ---------------------------------------------------------------------------
# Public training functions
# ---------------------------------------------------------------------------

def train_single(features_csv, config, output_dir):
    """
    Train on a single train / val split.

    Parameters
    ----------
    features_csv : str | Path   CSV produced by extract_features.py (with 'class' col)
    config       : dict         Full config dict
    output_dir   : str          Where to save pipeline.joblib and results.json

    Returns
    -------
    dict  val metrics
    """
    class_names = config["data"]["class_names"]
    seed        = config.get("seed", 42)
    split       = config["training"]["train_val_split"]

    X, y, _, feature_cols = _load_features(features_csv, class_names)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=split, random_state=seed
    )
    print(f"  Train: {len(X_train)} samples  |  Val: {len(X_val)} samples")

    pipeline = build_pipeline(config)
    estimator = _maybe_finetune(pipeline, X_train, y_train, config)

    y_pred = estimator.predict(X_val)
    y_prob = estimator.predict_proba(X_val)
    metrics = compute_metrics(y_val, y_pred, y_prob, class_names)

    print_metrics(metrics, class_names)
    result = _save_artefacts(estimator, feature_cols, metrics, output_dir, config)
    print(f"\n  Saved pipeline → {output_dir}/pipeline.joblib")
    return result


def train_kfold(features_csv, config, output_dir):
    """
    Train with N-fold cross-validation.

    Each fold is saved to output_dir/fold_N/.  A fold whose
    fold_complete.json marker already exists is skipped (resume support).
    A cv_results.json manifest is written to output_dir/ when all folds
    are done — this file is read by evaluate.py for ensemble inference.

    Parameters
    ----------
    features_csv : str | Path
    config       : dict
    output_dir   : str

    Returns
    -------
    dict  cv_results (same structure as the deep-learning cv_results.json)
    """
    class_names = config["data"]["class_names"]
    k           = config["training"]["k_folds"]
    seed        = config.get("seed", 42)

    X, y, _, feature_cols = _load_features(features_csv, class_names)

    skf          = KFold(n_splits=k, shuffle=True, random_state=seed)
    fold_results = []
    fold_paths   = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        fold_dir    = os.path.join(output_dir, f"fold_{fold_idx}")
        done_marker = os.path.join(fold_dir, "fold_complete.json")
        pipeline_path = os.path.join(fold_dir, "pipeline.joblib")

        # Resume support: skip already-finished folds
        if os.path.exists(done_marker):
            print(f"  Fold {fold_idx}/{k}: already complete — skipping.")
            with open(done_marker) as fh:
                fold_results.append(json.load(fh))
            fold_paths.append(pipeline_path)
            continue

        print(f"\n{'='*60}")
        print(f"  Fold {fold_idx}/{k}  |  train={len(train_idx)}  val={len(val_idx)}")
        print(f"{'='*60}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pipeline  = build_pipeline(config)
        estimator = _maybe_finetune(pipeline, X_train, y_train, config)

        y_pred  = estimator.predict(X_val)
        y_prob  = estimator.predict_proba(X_val)
        metrics = compute_metrics(y_val, y_pred, y_prob, class_names)

        print_metrics(metrics, class_names)

        fold_result = {
            "fold":         fold_idx,
            "val_accuracy": metrics["accuracy"],
            "val_f1_macro": metrics["f1_macro"],
            "metrics":      metrics,
        }
        _save_artefacts(estimator, feature_cols, metrics, fold_dir, config)
        with open(done_marker, "w") as fh:
            json.dump(fold_result, fh, indent=2)

        fold_results.append(fold_result)
        fold_paths.append(pipeline_path)

        print(f"\n  Fold {fold_idx} → Acc={metrics['accuracy']:.4f}  F1={metrics['f1_macro']:.4f}")

    # Cross-validation summary
    accs = [r["val_accuracy"] for r in fold_results]
    f1s  = [r["val_f1_macro"]  for r in fold_results]

    print(f"\n{'='*60}")
    print(f"  CV Summary ({k}-fold)")
    print(f"  Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  F1 Macro : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    # ------------------------------------------------------------------
    # Final model — retrained on ALL training data.
    # CV folds give unbiased performance estimates; this model is what
    # evaluate.py will actually use for test-set inference.
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Training final model on ALL data …")
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    final_pipeline  = build_pipeline(config)
    final_estimator = _maybe_finetune(final_pipeline, X, y, config)
    final_path      = os.path.join(final_dir, "pipeline.joblib")
    joblib.dump(final_estimator, final_path)
    print(f"  Final model → {final_path}")

    cv_results = {
        "experiment_name":     config["experiment_name"],
        "classifier":          config["classifier"]["type"],
        "k_folds":             k,
        "mean_val_accuracy":   float(np.mean(accs)),
        "std_val_accuracy":    float(np.std(accs)),
        "mean_val_f1_macro":   float(np.mean(f1s)),
        "std_val_f1_macro":    float(np.std(f1s)),
        "fold_results":        fold_results,
        "fold_pipeline_paths": fold_paths,
        "final_pipeline_path": final_path,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "cv_results.json"), "w") as fh:
        json.dump(cv_results, fh, indent=2)

    print(f"  cv_results.json → {output_dir}")

    return cv_results
