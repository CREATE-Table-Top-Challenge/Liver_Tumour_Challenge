"""
Task 1: Segmentation Evaluation Script
Run evaluation on validation or test data with your best model
"""
import argparse
import os
import glob
import json
import math
import warnings
import yaml
import logging
import zipfile
import numpy as np
import nibabel as nib
import torch
from pathlib import Path
from tqdm import tqdm

# Suppress MONAI's internal deprecation warning for get_mask_edges
# (always_return_as_numpy deprecated in 1.5, used internally by HausdorffDistanceMetric)
warnings.filterwarnings(
    "ignore",
    message=".*always_return_as_numpy.*",
    category=FutureWarning,
    module="monai",
)
# Suppress expected HD95 warning when a class has an all-zero prediction mask
# (NaN is handled gracefully via nanmean aggregation)
warnings.filterwarnings(
    "ignore",
    message=".*prediction of class.*is all 0.*",
    category=UserWarning,
    module="monai",
)
# Suppress PyTorch 2.9 deprecation warnings from MONAI's sliding window inference
# (MONAI will handle indexing updates in future versions; our code doesn't call these directly)
warnings.filterwarnings(
    "ignore",
    message=".*Using a non-tuple sequence for multidimensional indexing.*",
    category=UserWarning,
    module="monai.inferers",
)

from monai.data import decollate_batch, DataLoader, CacheDataset
from monai.data.meta_tensor import MetaTensor
from monai.inferers import sliding_window_inference

from src.base_model import build_model
from src.data_loader import load_data, get_data_loaders
from src.transforms import get_test_transforms, get_post_transforms


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Task 1: Segmentation Model Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model checkpoint
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./model_checkpoints/baseline_unet/best_model.pth',
        help='Path to model checkpoint'
    )
    
    # Configuration (optional, for data paths)
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    
    # Data paths (override config)
    parser.add_argument('--test_images', type=str, help='Path to test images directory')
    parser.add_argument('--test_labels', type=str, default=None, 
                       help='Path to test labels directory (optional; if provided, metrics will be computed)')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=None, help='Number of segmentation classes')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--class_names', type=str, nargs='+', default=None,
                       help='Class names (foreground only)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Output directory for evaluation results')
    parser.add_argument(
        '--group', type=int, default=None,
        help='Group number used to name the submission zip '
             '(e.g. 1 -> group1_task1_results.zip). '
             'Falls back to group_number in the config file.',
    )

    return parser.parse_args()


def setup_logging(output_dir: str):
    """Set up logging."""
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'evaluation.log')),
            logging.StreamHandler()
        ]
    )


def main():
    args = parse_args()
    
    # Load config if provided
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Override with command line args
    if args.test_images is None and 'data' in config and 'test_images' in config['data']:
        args.test_images = config['data']['test_images']
    if args.test_labels is None and 'data' in config and 'test_labels' in config['data']:
        args.test_labels = config['data']['test_labels']

    # Try to load model_config.json from checkpoint directory to get the exact architecture
    model_config_json = None
    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    model_config_path = os.path.join(checkpoint_dir, 'model_config.json')
    if os.path.exists(model_config_path):
        try:
            with open(model_config_path, 'r') as f:
                model_config_json = json.load(f)
        except Exception as e:
            logging.warning(f"Could not load model_config.json: {e}")

    # Resolve num_classes and class_names: CLI > model_config.json > config > fallback
    if args.num_classes is None:
        if model_config_json and 'num_classes' in model_config_json:
            args.num_classes = model_config_json['num_classes']
        else:
            model_cfg = config.get('model', {})
            args.num_classes = model_cfg.get('num_classes', 2)
    
    if args.class_names is None:
        model_cfg = config.get('model', {})
        args.class_names = model_cfg.get('class_names', ['liver', 'tumor'])

    # Resolve group number: CLI > config > None
    group_number = args.group if args.group is not None else config.get('group_number', None)

    # Set up logging
    setup_logging(args.output_dir)
    
    logging.info("=" * 60)
    logging.info("Task 1: Segmentation Model Evaluation")
    logging.info("=" * 60)
    logging.info(f"Checkpoint: {args.checkpoint}")
    logging.info(f"Test images: {args.test_images}")
    logging.info(f"Test labels: {args.test_labels}")
    logging.info(f"Number of classes: {args.num_classes}")
    logging.info(f"Class names: {args.class_names}")
    logging.info(f"Output directory: {args.output_dir}")
    if group_number is not None:
        logging.info(f"Group number: {group_number} -> will produce group{group_number}_task1_results.zip")
    if model_config_json:
        logging.info(f"Loaded model architecture from checkpoint: {model_config_json['arch_type']}")
    logging.info("=" * 60)
    
    # Validate paths
    if not os.path.exists(args.checkpoint):
        logging.error(f"Checkpoint not found: {args.checkpoint}")
        return
    
    if not args.test_images or not os.path.exists(args.test_images):
        logging.error(f"Test images not found: {args.test_images}")
        return
    
    has_labels = bool(args.test_labels and os.path.exists(args.test_labels))
    if not has_labels:
        logging.info("No test labels provided -- running inference only (no metrics).")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        logging.info(f"Using device: cuda (GPU: {torch.cuda.get_device_name(0)})")
    else:
        logging.warning("CUDA not available! Using CPU")
    
    # Load test data
    logging.info("\nLoading test data...")

    image_files = sorted(glob.glob(os.path.join(args.test_images, "*.nii.gz")))
    if has_labels:
        label_files = sorted(glob.glob(os.path.join(args.test_labels, "*.nii.gz")))
        val_files = [
            {"image": img, "label": lbl}
            for img, lbl in zip(image_files, label_files)
        ]
    else:
        val_files = [{"image": img} for img in image_files]
    
    logging.info(f"Found {len(val_files)} test samples")

    # Test transforms stop before CropForegroundd/SpatialPadd so Invertd only
    # undoes Spacingd + Orientationd, restoring predictions to the original voxel grid.
    test_transforms = get_test_transforms(allow_missing_keys=not has_labels)
    post_transforms = get_post_transforms(test_transforms)

    # Use group-based folder name if available, otherwise use a generic results name
    pred_dir_name = f'group{group_number}_task1_results' if group_number is not None else 'task1_segmentation_results'
    pred_dir = os.path.join(args.output_dir, pred_dir_name)
    os.makedirs(pred_dir, exist_ok=True)
    logging.info(f"Predictions will be saved to: {pred_dir}")

    val_dataset = CacheDataset(
        data=val_files,
        transform=test_transforms,
        cache_rate=0.5,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 for evaluation
        pin_memory=torch.cuda.is_available()
    )
    
    # Load model
    logging.info("\nLoading model...")
    
    # Use architecture from checkpoint's model_config.json if available; fallback to YAML config
    if model_config_json:
        arch_type = model_config_json['arch_type']
        arch_params = model_config_json.get('arch_params', {})
        # Merge checkpoint arch_params into config for build_model()
        if 'architecture' not in config:
            config['architecture'] = {}
        config['architecture']['type'] = arch_type
        config['architecture'][arch_type] = arch_params
    else:
        arch_type = config.get('architecture', {}).get('type', 'unet')
    
    logging.info(f"Architecture: {arch_type}")
    model = build_model(
        config=config,
        num_classes=args.num_classes,
        class_names=args.class_names,
        learning_rate=1e-4,  # not used during eval, required by base class
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'epoch' in checkpoint:
            logging.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        if 'best_metric' in checkpoint:
            logging.info(f"Best metric in checkpoint: {checkpoint['best_metric']:.4f}")
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    logging.info("Model loaded successfully")
    
    # Run evaluation
    logging.info("\n" + "=" * 60)
    logging.info("Running evaluation...")
    logging.info("=" * 60)

    val_metrics = {}
    total_samples = 0
    sample_idx = 0  # Track overall sample index
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc='Evaluating')):
            # Move batch to device first (before saving predictions)
            batch_for_eval = {k: v.to(device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Validation step (only when GT labels are available)
            if has_labels:
                metrics = model.validation_step(batch_for_eval)
            
            # Sliding-window inference — roi_size and sw_batch_size are read
            # from the model so they always match the trained architecture.
            images = batch_for_eval['image']
            outputs = sliding_window_inference(
                images, model.roi_size, model.sw_batch_size, model.net
            )

            # Argmax before Invertd: collapse to single-channel label map
            # (nearest-neighbour inversion of an integer map is correct)
            preds_argmax = torch.argmax(outputs, dim=1, keepdim=True).cpu()

            # Decollate WITHOUT pred, then assign pred as MetaTensor carrying
            # the image's spatial metadata so Invertd can invert it correctly.
            batch_list = decollate_batch(batch)

            for i, item in enumerate(batch_list):
                current_idx = sample_idx + i
                base_name = (os.path.basename(val_files[current_idx]['image'])
                             if current_idx < len(val_files)
                             else f"prediction_{current_idx:03d}.nii.gz")

                pred_tensor = preds_argmax[i]
                if hasattr(item["image"], 'meta'):
                    item["pred"] = MetaTensor(pred_tensor, meta=item["image"].meta)
                else:
                    item["pred"] = MetaTensor(pred_tensor)

                # Invert Spacing + Orientation -> original voxel grid
                item = post_transforms(item)

                pred = item['pred']
                pred_mask = (pred.cpu().numpy().squeeze() if torch.is_tensor(pred)
                             else np.asarray(pred).squeeze())

                # Load affine directly from the original NIfTI (most reliable)
                orig_img = nib.load(val_files[current_idx]['image']) if current_idx < len(val_files) else None
                orig_affine = orig_img.affine if orig_img is not None else np.eye(4)

                pred_nii = nib.Nifti1Image(pred_mask.astype(np.uint8), orig_affine)
                nib.save(pred_nii, os.path.join(pred_dir, base_name))
            
            # Update sample index
            sample_idx += len(batch_list)
            
            # Accumulate metrics
            batch_size = batch_for_eval['image'].size(0)
            total_samples += batch_size

            if has_labels:
                for k, v in metrics.items():
                    if k not in val_metrics:
                        val_metrics[k] = 0
                    val_metrics[k] += v * batch_size
    
    logging.info(f"\nPredictions saved to: {pred_dir}")
    logging.info(f"Total predictions saved: {sample_idx}")

    # Package submission zip when a group number is known
    if group_number is not None:
        zip_path = Path(args.output_dir) / f'group{group_number}_task1_results.zip'
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            for mask_file in sorted(Path(pred_dir).glob('*.nii.gz')):
                zf.write(mask_file, arcname=f'{pred_dir_name}/{mask_file.name}')
        logging.info(f"[OK] Submission zip created: {zip_path}")

    results = {
        'checkpoint': args.checkpoint,
        'num_samples': total_samples,
    }

    if has_labels:
        # Calculate averages
        for k in val_metrics:
            val_metrics[k] /= total_samples

        # Get per-class metrics
        class_metrics = model.on_validation_epoch_end()

        logging.info("\n" + "=" * 60)
        logging.info("Evaluation Results")
        logging.info("=" * 60)
        logging.info(f"Validation Loss: {val_metrics.get('val_loss', 0):.4f}")
        logging.info("")

        for class_name in args.class_names:
            dice_key = f'val_dice_{class_name}'
            hd95_key = f'val_hd95_{class_name}'
            if dice_key in class_metrics:
                logging.info(f"{class_name.capitalize()}:")
                logging.info(f"  Dice Score: {class_metrics[dice_key]:.4f}")
                if hd95_key in class_metrics and not math.isnan(class_metrics[hd95_key]):
                    logging.info(f"  HD95:       {class_metrics[hd95_key]:.4f} mm")

        dice_scores = [class_metrics[f'val_dice_{name}'] for name in args.class_names
                       if f'val_dice_{name}' in class_metrics]
        hd95_scores = [class_metrics[f'val_hd95_{name}'] for name in args.class_names
                       if f'val_hd95_{name}' in class_metrics]

        mean_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0
        valid_hd95 = [v for v in hd95_scores if not math.isnan(v)]
        mean_hd95 = sum(valid_hd95) / len(valid_hd95) if valid_hd95 else float('nan')

        if dice_scores:
            logging.info("")
            logging.info("Overall:")
            logging.info(f"  Mean Dice Score: {mean_dice:.4f}")
            if not math.isnan(mean_hd95):
                logging.info(f"  Mean HD95:       {mean_hd95:.4f} mm")

        results.update({
            'validation_loss': val_metrics.get('val_loss', 0),
            'class_metrics': class_metrics,
            'mean_dice': mean_dice,
            'mean_hd95': mean_hd95,
        })

    logging.info("=" * 60)

    results_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info(f"\nResults saved to: {results_file}")
    logging.info("\nEvaluation complete!")


if __name__ == '__main__':
    main()
