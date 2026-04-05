"""
Task 2: Classification Training Script
Unified CLI entry point for training tumor response classification models
"""
import argparse
import os
import yaml
import torch
import torch.nn as nn
import logging
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from src.base_model import build_model, _ARCH_REGISTRY
from src.data_loader import load_data, get_data_loaders, split_dataset, create_cv_folds
from src.transforms import get_train_transforms, get_val_transforms
from src.trainer import Trainer
from src.metrics import ClassificationMetrics
from torch.utils.data import Dataset


class TransformSubset(Dataset):
    """Wrap a Subset so train and val can carry different transforms.

    Each worker process receives its own pickled copy, so swapping the parent
    dataset's transforms inside __getitem__ is safe (no cross-worker races).
    """

    def __init__(self, subset, transforms):
        self.subset = subset
        self.transforms = transforms

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Temporarily disable the parent dataset's transforms so we get raw data,
        # then apply our own transforms.
        ds = self.subset.dataset
        saved = ds.transforms
        ds.transforms = None
        item = self.subset[idx]
        ds.transforms = saved
        if self.transforms is not None:
            item["image"] = self.transforms(item["image"])
        return item


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Task 2: Liver Tumors Classification Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, help='Path to training data directory')
    parser.add_argument('--labels_csv', type=str, help='Path to labels CSV file')
    parser.add_argument('--val_dir', type=str, help='Path to validation data directory (optional)')
    parser.add_argument('--val_labels_csv', type=str, help='Path to validation labels CSV (required if val_dir is specified)')
    
    # Model parameters
    parser.add_argument('--model_type', type=str,
                       choices=list(_ARCH_REGISTRY), help='Model architecture')
    parser.add_argument('--num_classes', type=int, help='Number of classes')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, help='Weight decay')
    
    # Training parameters
    parser.add_argument('--max_epochs', type=int, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, help='Number of data loading workers')
    parser.add_argument('--val_interval', type=int, help='Validation every N epochs')
    parser.add_argument('--patience', type=int, help='Early stopping patience')
    
    # Cross-validation
    parser.add_argument('--k_folds', type=int, help='Number of folds for k-fold CV (0=no CV)')
    parser.add_argument('--train_val_split', type=float, 
                       help='Train/val split ratio if no separate val_dir')
    
    # Output
    parser.add_argument('--output_dir', type=str, 
                       help='Output directory for checkpoints')
    parser.add_argument('--experiment_name', type=str,
                       help='Experiment name for logging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Resume training
    parser.add_argument('--resume_from', type=str, help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_config_with_args(config: dict, args: argparse.Namespace) -> argparse.Namespace:
    """
    Merge YAML config with command line arguments.
    Command line arguments take precedence over config file.
    
    Args:
        config: Configuration dictionary from YAML
        args: Command line arguments
        
    Returns:
        Updated arguments namespace
    """
    # Model parameters
    if 'model' in config:
        if args.model_type is None and 'model_type' in config['model']:
            args.model_type = config['model']['model_type']
        if args.num_classes is None and 'num_classes' in config['model']:
            args.num_classes = config['model']['num_classes']
        if args.learning_rate is None and 'learning_rate' in config['model']:
            args.learning_rate = float(config['model']['learning_rate'])
        if args.weight_decay is None and 'weight_decay' in config['model']:
            args.weight_decay = float(config['model']['weight_decay'])
        if not args.pretrained and 'pretrained' in config['model']:
            args.pretrained = config['model']['pretrained']
    
    # Training parameters
    if 'training' in config:
        if args.max_epochs is None and 'epochs' in config['training']:
            args.max_epochs = config['training']['epochs']
        if args.batch_size is None and 'batch_size' in config['training']:
            args.batch_size = config['training']['batch_size']
        if args.num_workers is None and 'num_workers' in config['training']:
            args.num_workers = config['training']['num_workers']
        if args.val_interval is None and 'validation_interval' in config['training']:
            args.val_interval = config['training']['validation_interval']
        if args.patience is None and 'patience' in config['training']:
            args.patience = config['training']['patience']
        if args.k_folds is None and 'k_folds' in config['training']:
            args.k_folds = config['training']['k_folds']
    
    # Data paths
    if 'data' in config:
        if args.data_dir is None and 'train_dir' in config['data']:
            args.data_dir = config['data']['train_dir']
        if args.labels_csv is None and 'labels_csv' in config['data']:
            args.labels_csv = config['data']['labels_csv']
        if args.val_dir is None and 'val_dir' in config['data']:
            args.val_dir = config['data']['val_dir']
        if args.val_labels_csv is None and 'val_labels_csv' in config['data']:
            args.val_labels_csv = config['data']['val_labels_csv']
        if args.train_val_split is None and 'train_val_split' in config['data']:
            args.train_val_split = config['data']['train_val_split']
    
    # Output
    if 'output' in config:
        if args.output_dir is None and 'checkpoint_dir' in config['output']:
            args.output_dir = config['output']['checkpoint_dir']
    if args.experiment_name is None and 'experiment_name' in config:
        args.experiment_name = config['experiment_name']
    
    # Set defaults if still None
    if args.output_dir is None:
        args.output_dir = './model_checkpoints'
    if args.experiment_name is None:
        args.experiment_name = 'classification_exp'
    if args.model_type is None:
        args.model_type = 'resnet18'
    if args.num_classes is None:
        args.num_classes = 3
    if args.learning_rate is None:
        args.learning_rate = 1e-4
    if args.weight_decay is None:
        args.weight_decay = 1e-4
    if args.max_epochs is None:
        args.max_epochs = 50
    if args.batch_size is None:
        args.batch_size = 16
    if args.num_workers is None:
        args.num_workers = 4
    if args.val_interval is None:
        args.val_interval = 1
    if args.patience is None:
        args.patience = 10
    if args.k_folds is None:
        args.k_folds = 0
    if args.train_val_split is None:
        args.train_val_split = 0.8
    
    return args


def setup_logging(output_dir: str, experiment_name: str):
    """
    Set up logging to both file and console.
    
    Args:
        output_dir: Directory to save log file
        experiment_name: Name of the experiment
    """
    log_dir = os.path.join(output_dir, experiment_name, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )


def create_model(args):
    """
    Create classification model from arguments.

    Args:
        args: Command line arguments (must have model_type, num_classes, _config).

    Returns:
        Model instance
    """
    return build_model(
        config=getattr(args, '_config', {}),
        num_classes=args.num_classes,
        model_type=args.model_type,
    )


def run_cv_training(args, full_dataset, device):
    """
    Run stratified k-fold cross-validation.

    Each fold trains a freshly initialised model for ``max_epochs`` epochs
    (with early stopping) and saves checkpoints under
    ``output_dir/experiment_name/fold_K/``.

    After all folds finish, mean ± std of accuracy and F1 are printed and
    logged. The fold with the highest validation accuracy is flagged as the
    best fold.

    Args:
        args:         Merged argument namespace.
        full_dataset: ROIDataset loaded without transforms.
        device:       Torch device string ('cuda' or 'cpu').
    """
    import numpy as np
    import json

    logging.info(f"Starting {args.k_folds}-fold stratified cross-validation")

    enable_stratified_split = args._config.get("data", {}).get("enable_stratified_split", True)
    folds = create_cv_folds(full_dataset, n_splits=args.k_folds, random_seed=args.seed, enable_stratified_split=enable_stratified_split)

    spatial_size = args._config.get("training", {}).get("spatial_size", None)
    enable_augmentation = args._config.get("training", {}).get("enable_augmentation", True)
    train_transforms = get_train_transforms(spatial_size=spatial_size, enable_augmentation=enable_augmentation)
    val_transforms   = get_val_transforms(spatial_size=spatial_size)

    fold_results = []   # list of dicts with best val metrics per fold

    for fold_idx, (train_subset, val_subset) in enumerate(folds):
        fold_num = fold_idx + 1
        fold_output_dir = os.path.join(
            args.output_dir, args.experiment_name, f"fold_{fold_num}"
        )
        fold_complete_path = os.path.join(fold_output_dir, 'fold_complete.json')

        # ---- Skip already-finished folds (resume support) ----
        if os.path.isfile(fold_complete_path):
            with open(fold_complete_path, encoding='utf-8') as f:
                prev = json.load(f)
            logging.info("="*60)
            logging.info(f"FOLD {fold_num}/{args.k_folds} [SKIPPED - already complete]")
            logging.info(
                f"  best acc: {prev['best_val_acc']:.4f} at epoch {prev['best_epoch']}"
            )
            logging.info("="*60)
            fold_results.append({
                'fold':            fold_num,
                'best_acc':        prev['best_val_acc'],
                'best_epoch':      prev['best_epoch'],
                'best_model_path': prev['best_model_path'],
            })
            continue

        logging.info("="*60)
        logging.info(f"FOLD {fold_num}/{args.k_folds}")
        logging.info("="*60)

        # Wrap with transforms
        train_ds = TransformSubset(train_subset, train_transforms)
        val_ds   = TransformSubset(val_subset,   val_transforms)

        train_loader, val_loader = get_data_loaders(
            train_ds, val_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # Fresh model for every fold
        model = create_model(args)

        # fold_output_dir already set above (and used for fold_complete_path check)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir=fold_output_dir,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            val_interval=args.val_interval,
            patience=args.patience,
        )

        trainer.train()

        fold_result = {
            'fold':            fold_num,
            'best_acc':        trainer.best_val_acc,
            'best_epoch':      trainer.best_epoch,
            'best_model_path': trainer.best_model_path,
        }
        fold_results.append(fold_result)

        # ---- Write completion marker so this fold is skipped on resume ----
        completion_record = {
            'fold':            fold_num,
            'best_val_acc':    trainer.best_val_acc,
            'best_epoch':      trainer.best_epoch,
            'best_model_path': trainer.best_model_path,
        }
        os.makedirs(fold_output_dir, exist_ok=True)
        with open(fold_complete_path, 'w') as f:
            json.dump(completion_record, f, indent=2)
        logging.info(f"[OK] Fold {fold_num} complete marker written: {fold_complete_path}")

        logging.info(
            f"Fold {fold_num} complete - "
            f"best acc: {trainer.best_val_acc:.4f} at epoch {trainer.best_epoch}"
        )

    # ---- Summary ----
    accs = [r['best_acc'] for r in fold_results]
    logging.info("="*60)
    logging.info(f"Cross-validation summary ({args.k_folds} folds)")
    logging.info("="*60)
    for r in fold_results:
        logging.info(
            f"  Fold {r['fold']}: best val acc = {r['best_acc']:.4f} "
            f"(epoch {r['best_epoch']})"
        )
    logging.info(
        f"  Mean acc: {np.mean(accs):.4f}  |  "
        f"Std: {np.std(accs):.4f}  |  "
        f"Min: {np.min(accs):.4f}  |  "
        f"Max: {np.max(accs):.4f}"
    )
    best = max(fold_results, key=lambda r: r['best_acc'])
    logging.info(
        f"  Best fold: {best['fold']} "
        f"(acc={best['best_acc']:.4f}, model: {best['best_model_path']})"
    )
    logging.info("="*60)

    # ---- Save cv_results.json manifest for ensemble inference ----
    # Strategy: average the softmax probabilities from all K fold models
    # (probability averaging == soft voting) — standard competition best practice.
    #
    # All fold models are copied into  <cv_dir>/models/fold_K/best_model.pth
    # and the manifest stores RELATIVE paths so the whole models/ directory is
    # portable — just copy it into the Docker image as /app/models/ and the
    # inferer will find and ensemble all fold models automatically.
    import json, shutil

    cv_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(cv_dir, exist_ok=True)

    fold_entries = []
    for r in fold_results:
        dst_dir  = os.path.join(cv_dir, 'models', f'fold_{r["fold"]}')
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, 'best_model.pth')
        if r['best_model_path'] and os.path.isfile(r['best_model_path']):
            shutil.copy2(r['best_model_path'], dst_path)
            logging.info(f"[OK] Fold {r['fold']} model -> {dst_path}")
        rel_path = os.path.join('models', f'fold_{r["fold"]}', 'best_model.pth')
        fold_entries.append({
            'fold':         r['fold'],
            'best_val_acc': r['best_acc'],
            'best_epoch':   r['best_epoch'],
            'model_path':   rel_path,   # relative to the cv_results.json directory
        })

    cv_manifest = {
        'k_folds':   args.k_folds,
        'mean_acc':  float(np.mean(accs)),
        'std_acc':   float(np.std(accs)),
        'best_fold': int(best['fold']),
        'folds':     fold_entries,
    }

    manifest_path = os.path.join(cv_dir, 'cv_results.json')
    with open(manifest_path, 'w') as f:
        json.dump(cv_manifest, f, indent=2)
    logging.info(f"[OK] Saved CV manifest: {manifest_path}")

    # Single-model fallback: copy best-fold model to models/best_model.pth
    # (used when the inferer cannot find the manifest, or for quick tests).
    best_fallback = os.path.join(cv_dir, 'models', 'best_model.pth')
    best_src = os.path.join(cv_dir, 'models', f'fold_{best["fold"]}', 'best_model.pth')
    if os.path.isfile(best_src):
        shutil.copy2(best_src, best_fallback)
        logging.info(
            f"[OK] Best-fold fallback model (fold {best['fold']}) -> {best_fallback}"
        )

    logging.info(
        f"[*] To build the Docker submission, copy "
        f"{cv_dir}/models/  ->  task2_classification/models/ "
        f"and {cv_dir}/cv_results.json  ->  task2_classification/models/cv_results.json"
    )
    logging.info(
        f"[*] Inference mode will be: ENSEMBLE of all {args.k_folds} fold models "
        f"(probability averaging) for best competition accuracy."
    )


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        args = merge_config_with_args(config, args)
        args._config = config   # preserved for build_model() arch-specific params
    else:
        args._config = {}
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Setup logging
    setup_logging(args.output_dir, args.experiment_name)
    
    # Log configuration
    logging.info("="*60)
    logging.info("Task 2: Liver Tumors Classification Training")
    logging.info("="*60)
    logging.info(f"Configuration:")
    for key, value in vars(args).items():
        logging.info(f"  {key}: {value}")
    logging.info("="*60)
    
    # Validate required arguments
    if not args.data_dir:
        raise ValueError("Missing required argument: --data_dir or config data.train_dir")
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    
    if device == 'cuda':
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load data (always without transforms at this stage)
    logging.info("Loading data...")
    if not args.labels_csv:
        raise ValueError("labels_csv is required. Provide via --labels_csv or config file")
    full_dataset = load_data(args.data_dir, args.labels_csv, transforms=None)

    # ------------------------------------------------------------------ #
    # Cross-validation path
    # ------------------------------------------------------------------ #
    if args.k_folds and args.k_folds > 1:
        run_cv_training(args, full_dataset, device)
        return

    # ------------------------------------------------------------------ #
    # Standard single train/val split path
    # ------------------------------------------------------------------ #
    # Load or create validation data
    if args.val_dir:
        # Separate train and val directories
        logging.info(f"Loading separate validation data from {args.val_dir}")
        if not args.val_labels_csv:
            raise ValueError("val_labels_csv is required when using val_dir")
        spatial_size = args._config.get("training", {}).get("spatial_size", None)
        enable_augmentation = args._config.get("training", {}).get("enable_augmentation", True)
        train_transforms = get_train_transforms(spatial_size=spatial_size, enable_augmentation=enable_augmentation)
        val_transforms = get_val_transforms(spatial_size=spatial_size)

        train_dataset = load_data(args.data_dir, args.labels_csv, transforms=train_transforms)
        val_dataset   = load_data(args.val_dir, args.val_labels_csv, transforms=val_transforms)
    else:
        # Stratified split
        logging.info(f"Splitting training data with ratio {args.train_val_split}")
        enable_stratified_split = args._config.get("data", {}).get("enable_stratified_split", True)
        train_subset, val_subset = split_dataset(
            full_dataset,
            train_ratio=args.train_val_split,
            random_seed=args.seed,
            enable_stratified_split=enable_stratified_split,
        )
        spatial_size = args._config.get("training", {}).get("spatial_size", None)
        enable_augmentation = args._config.get("training", {}).get("enable_augmentation", True)
        train_transforms = get_train_transforms(spatial_size=spatial_size, enable_augmentation=enable_augmentation)
        val_transforms   = get_val_transforms(spatial_size=spatial_size)
        train_dataset = TransformSubset(train_subset, train_transforms)
        val_dataset   = TransformSubset(val_subset,   val_transforms)
    
    # Create data loaders
    logging.info("Creating data loaders...")
    train_loader, val_loader = get_data_loaders(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    logging.info(f"Training batches: {len(train_loader)}")
    logging.info(f"Validation batches: {len(val_loader)}")
    
    # Create model
    logging.info("Initializing model...")
    model = create_model(args)
    logging.info(f"Model: {args.model_type}, Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Set up trainer
    logging.info("Setting up trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=os.path.join(args.output_dir, args.experiment_name),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        val_interval=args.val_interval,
        patience=args.patience
    )
    
    # Load checkpoint if specified (for resuming training)
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
        logging.info(f"Resumed training from {args.resume_from}")
    
    # Train model
    logging.info("Starting training...")
    logging.info("="*60)
    trainer.train()
    
    logging.info("="*60)
    logging.info(f"Training complete!")
    logging.info(f"Best model saved to: {trainer.best_model_path}")
    logging.info("="*60)


if __name__ == '__main__':
    main()
