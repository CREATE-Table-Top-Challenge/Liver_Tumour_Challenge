#!/usr/bin/env python
"""
Step 2 — Training
==================
Train a scikit-learn classifier pipeline on the radiomics features produced
by extract_features.py.

Usage — single train / val split  (default)
-------------------------------------------
    python train.py \\
        --config config.yaml \\
        --features features/train_features.csv

Usage — k-fold cross-validation
---------------------------------
    python train.py \\
        --config config.yaml \\
        --features features/train_features.csv \\
        --k-folds 5

CLI overrides  (all optional — override the YAML without editing it)
----------------------------------------------------------------------
    --classifier   random_forest | svm | logistic_regression | xgboost | mlp
    --k-folds      0 | 1 | 5
    --output-dir   ./model_checkpoints/my_experiment
    --seed         42
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

# Ensure the package root is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))
from src.trainer import train_kfold, train_single


def _parse_args():
    p = argparse.ArgumentParser(description="Radiomics baseline — training")
    p.add_argument("--config",      default="config.yaml",
                   help="Path to config.yaml  (default: config.yaml)")
    p.add_argument("--features",    required=True,
                   help="Train feature CSV produced by extract_features.py")
    p.add_argument("--output-dir",  default=None,
                   help="Override output directory  (default: from config)")
    p.add_argument("--classifier",  default=None,
                   choices=["random_forest", "svm", "logistic_regression",
                             "xgboost", "mlp"],
                   help="Override classifier type")
    p.add_argument("--k-folds",  type=int, default=None,
                   help="Override k_folds  (0/1 = single split, N>1 = CV)")
    p.add_argument("--seed",     type=int, default=None,
                   help="Override random seed")
    return p.parse_args()


def main():
    args = _parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    # Apply optional CLI overrides
    if args.classifier is not None:
        config["classifier"]["type"] = args.classifier
    if args.k_folds is not None:
        config["training"]["k_folds"] = args.k_folds
    if args.seed is not None:
        config["seed"] = args.seed

    checkpoint_base = args.output_dir or config["output"]["checkpoint_dir"]
    output_dir      = os.path.join(checkpoint_base, config["experiment_name"])

    print("=" * 60)
    print(f"  Experiment  : {config['experiment_name']}")
    print(f"  Classifier  : {config['classifier']['type']}")
    print(f"  k_folds     : {config['training']['k_folds']}")
    print(f"  Features    : {args.features}")
    print(f"  Output dir  : {output_dir}")
    print("=" * 60)

    k = config["training"]["k_folds"]
    if k > 1:
        print(f"\nStarting {k}-fold cross-validation …\n")
        train_kfold(args.features, config, output_dir)
    else:
        ratio = int(config["training"]["train_val_split"] * 100)
        print(f"\nStarting single split ({ratio}/{100 - ratio}) …\n")
        train_single(args.features, config, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
