"""
Task 2: Liver Tumour Classification -- Evaluate / Predict

Entry-point CLI for running inference on preprocessed test ROIs.
Produces predictions.csv for submission to the organizer.

Usage
-----
# 5-fold ensemble (recommended)
python evaluate.py \\
    --input  data/processed/test \\
    --output results/ \\
    --models-dir model_checkpoints/baseline_resnet18/

# Single checkpoint
python evaluate.py \\
    --input  data/processed/test \\
    --output results/ \\
    --model  model_checkpoints/baseline_resnet18/model_checkpoints/best_model.pth

Output: results/predictions.csv
    patient_id, predicted_class, prob_<class0>, prob_<class1>, ...
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Make src/ importable without installing the package
_src_dir = Path(__file__).resolve().parent / 'src'
sys.path.insert(0, str(_src_dir))

from inferer import (
    setup_logging,
    load_model,
    load_ensemble_models,
    load_roi_scan,
    preprocess_roi,
    predict_roi,
    predict_ensemble,
    save_csv,
)
from transforms import get_val_transforms


def parse_args():
    p = argparse.ArgumentParser(
        description='Task 2: Run inference on test ROIs and produce predictions.csv.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--config', default=None,
        help='Path to training config YAML (e.g. baseline_config.yaml). '
             'If not provided, you must specify --model_type, --num_classes, and --classes as CLI arguments.',
    )
    p.add_argument(
        '--input', required=True,
        help='Directory of preprocessed test .npy ROI files '
             '(output of prepare_dataset_for_task2.py --test-mode).',
    )
    p.add_argument(
        '--output', required=True,
        help='Directory where predictions.csv will be written.',
    )
    p.add_argument(
        '--models_dir', default=None,
        help='Directory containing cv_results.json (ensemble) or best_model.pth (single model). '
             'Script automatically checks nested model_checkpoints/ subfolder if present. '
             'cv_results.json takes priority when both are present.',
    )
    p.add_argument(
        '--model', default=None,
        help='Explicit path to a single .pth checkpoint (overrides --models_dir).',
    )
    p.add_argument(
        '--model_type', default=None,
        help='Model architecture identifier (required if --config not provided; default: resnet18 from config).',
    )
    p.add_argument(
        '--num_classes', type=int, default=None,
        help='Number of output classes (required if --config not provided; default: 5 from config).',
    )
    p.add_argument(
        '--classes', nargs='+', default=None,
        help='Class names in label-index order (required if --config not provided; default: from config model.class_names).',
    )
    p.add_argument(
        '--device', default=None,
        help='cuda or cpu (auto-detected when omitted).',
    )
    p.add_argument(
        '--group', type=int, default=None,
        help='Group number used to name the output CSV (e.g. 1 -> group1_task2_results.csv). '
             'Falls back to group_number in the config file.',
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ---- Validate required arguments ----
    if not args.config:
        # Config file not provided: all model settings must be passed as CLI args
        missing_args = []
        if args.model_type is None:
            missing_args.append('--model-type')
        if args.num_classes is None:
            missing_args.append('--num-classes')
        if args.classes is None:
            missing_args.append('--classes')
        
        if missing_args:
            print(
                f"ERROR: Config file not provided (--config). "
                f"The following arguments are required: {', '.join(missing_args)}\n"
                f"Either provide a config file with --config <path> OR all required CLI arguments.",
                file=sys.stderr
            )
            sys.exit(1)

    # ---- Resolve settings (config -> CLI override) ----
    cfg = {}
    if args.config:
        with open(args.config, encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}

    model_cfg  = cfg.get('model', {})
    output_cfg = cfg.get('output', {})

    model_type  = args.model_type  or model_cfg.get('model_type',  'resnet18')
    num_classes = args.num_classes or model_cfg.get('num_classes', 5)
    class_names = args.classes     or model_cfg.get('class_names', [f'class_{i}' for i in range(num_classes)])
    group_number = args.group or cfg.get('group_number', None)

    # Pad / truncate class_names to num_classes
    if len(class_names) != num_classes:
        class_names = (class_names + [f'class_{i}' for i in range(num_classes)])[:num_classes]

    # If models_dir not given on CLI, fall back to config's checkpoint_dir / experiment
    # (cv_results.json lives at that level, not inside models/)
    models_dir_arg = args.models_dir
    if models_dir_arg is None and not args.model and 'checkpoint_dir' in output_cfg:
        experiment     = cfg.get('experiment_name', 'experiment')
        models_dir_arg = str(Path(output_cfg['checkpoint_dir']) / experiment)

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    setup_logging(output_dir)

    logging.info("=" * 60)
    logging.info("Task 2: Liver Tumour Classification -- Inference")
    logging.info("=" * 60)
    logging.info(f"Model type  : {model_type}")
    logging.info(f"Num classes : {num_classes}")
    logging.info(f"Classes     : {class_names}")

    if not input_dir.exists():
        logging.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device      : {device}")
    if device == 'cuda':
        logging.info(f"GPU         : {torch.cuda.get_device_name(0)}")

    # ---- Load model(s) ----
    if args.model:
        logging.info(f"Mode: SINGLE MODEL  ({args.model})")
        models       = [load_model(args.model, model_type, num_classes, device)]
        use_ensemble = False
    elif models_dir_arg:
        models_dir       = Path(models_dir_arg)
        cv_manifest_path = models_dir / 'cv_results.json'
        single_ckpt      = models_dir / 'best_model.pth'
        
        # Check for nested model_checkpoints directory (created by trainer)
        nested_dir = models_dir / 'model_checkpoints'
        if nested_dir.exists():
            cv_manifest_nested = nested_dir / 'cv_results.json'
            single_ckpt_nested  = nested_dir / 'best_model.pth'
            if cv_manifest_nested.exists():
                cv_manifest_path = cv_manifest_nested
            if single_ckpt_nested.exists():
                single_ckpt = single_ckpt_nested
        
        if cv_manifest_path.exists():
            logging.info(f"Mode: ENSEMBLE  ({cv_manifest_path})")
            models       = load_ensemble_models(str(cv_manifest_path), model_type, num_classes, device)
            use_ensemble = True
        elif single_ckpt.exists():
            logging.info(f"Mode: SINGLE MODEL  ({single_ckpt})")
            models       = [load_model(str(single_ckpt), model_type, num_classes, device)]
            use_ensemble = False
        else:
            logging.error(f"Neither cv_results.json nor best_model.pth found in {models_dir} or {nested_dir}.")
            sys.exit(1)
    else:
        logging.error("Provide --model <path> or --models-dir <dir> (or set output.checkpoint_dir in --config).")
        sys.exit(1)

    training_cfg = cfg.get('training', {})
    spatial_size = training_cfg.get('spatial_size', None)
    transforms = get_val_transforms(spatial_size=spatial_size)

    # ---- Collect test scans ----
    scan_files = []
    for pattern in ('*.npy', '*.nii.gz', '*.nii'):
        scan_files.extend(sorted(input_dir.glob(pattern)))

    if not scan_files:
        logging.error(f"No scan files found in {input_dir}")
        sys.exit(1)

    logging.info(f"Found {len(scan_files)} scan(s) to process")
    logging.info("=" * 60)

    # ---- Run inference ----
    rows         = []
    class_counts = {name: 0 for name in class_names}

    for idx, scan_path in enumerate(scan_files, 1):
        scan_path  = Path(scan_path)
        patient_id = scan_path.stem.replace('.nii', '')   # handles .nii.gz double ext
        logging.info(f"[{idx}/{len(scan_files)}] {patient_id}")

        try:
            roi_data   = load_roi_scan(str(scan_path))
            roi_tensor = preprocess_roi(roi_data, transforms, device)

            if use_ensemble:
                pred_idx, probs = predict_ensemble(models, roi_tensor)
            else:
                pred_idx, probs = predict_roi(models[0], roi_tensor)

            pred_name = class_names[pred_idx]
            class_counts[pred_name] += 1
            logging.info(
                f"  -> {pred_name}  "
                + "  ".join(f"{class_names[i]}:{probs[i]:.3f}" for i in range(len(class_names)))
            )

            row = {'patient_id': patient_id, 'predicted_class': pred_name}
            for i, name in enumerate(class_names):
                row[f'prob_{name.replace(" ", "_")}'] = f'{probs[i]:.6f}'
            rows.append(row)

        except Exception as e:
            logging.error(f"  Error on {patient_id}: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)

    # ---- Write CSV ----
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_stem = f'group{group_number}_task2_results' if group_number is not None else 'predictions'
    csv_path = output_dir / f'{csv_stem}.csv'
    save_csv(rows, class_names, csv_path)

    logging.info("=" * 60)
    logging.info("Prediction Summary:")
    for name, count in class_counts.items():
        pct = 100 * count / len(rows) if rows else 0
        logging.info(f"  {name}: {count}  ({pct:.1f}%)")
    logging.info("=" * 60)
    logging.info(f"[Done] {len(rows)} cases  ->  {csv_path}")
    logging.info("Share predictions.csv with the organizer for scoring.")
    logging.info("=" * 60)


if __name__ == '__main__':
    main()
