"""
Task 1: Segmentation Training Script
Unified CLI entry point for training segmentation models (UNet, SegResNet, SwinUNETR)
"""
import argparse
import json
import os
import warnings
import yaml
import torch
import logging
from pathlib import Path

# Suppress MONAI's internal deprecation warning for get_mask_edges
# (always_return_as_numpy deprecated in 1.5, used internally by HausdorffDistanceMetric)
warnings.filterwarnings(
    "ignore",
    message=".*always_return_as_numpy.*",
    category=FutureWarning,
    module="monai",
)
# Suppress expected HD95 warning when a class has an all-zero prediction mask
# (NaN is handled gracefully via torch.nanmean in on_validation_epoch_end)
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

import monai

from src.base_model import build_model
from src.trainer import Trainer
from src.data_loader import load_data, load_data_random, get_data_loaders
from src.transforms import get_data_transforms


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Task 1: Liver Tumor Segmentation Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    
    # Data paths
    parser.add_argument('--train_images', type=str, help='Path to training images directory')
    parser.add_argument('--train_labels', type=str, help='Path to training labels directory')
    parser.add_argument('--val_images', type=str, help='Path to validation images directory')
    parser.add_argument('--val_labels', type=str, help='Path to validation labels directory')
    parser.add_argument('--val_fraction', type=float, default=None,
                        help='Fraction of training data to use as validation when '
                             '--val_images / --val_labels are not provided (default: 0.2)')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=None, help='Number of segmentation classes')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay')
    parser.add_argument('--class_names', type=str, nargs='+', default=None, help='Class names (foreground only)')
    parser.add_argument('--compute_hd95', type=lambda x: x.lower() in ('true', '1', 'yes'), default=None, 
                        help='Compute HD95 metric during validation (default: True; set to False/0/no to disable)')
    
    # Training parameters
    parser.add_argument('--max_epochs', type=int, default=None, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of data loading workers')
    parser.add_argument('--val_interval', type=int, default=None, help='Validation every N epochs')
    parser.add_argument('--patience', type=int, default=None, help='Early stopping patience')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for checkpoints')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    
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
        if args.num_classes is None and 'num_classes' in config['model']:
            args.num_classes = config['model']['num_classes']
        if args.learning_rate is None and 'learning_rate' in config['model']:
            args.learning_rate = float(config['model']['learning_rate'])
        if args.weight_decay is None and 'weight_decay' in config['model']:
            args.weight_decay = float(config['model']['weight_decay'])
        if args.class_names is None and 'class_names' in config['model']:
            args.class_names = config['model']['class_names']
        if args.compute_hd95 is None and 'compute_hd95' in config['model']:
            args.compute_hd95 = config['model']['compute_hd95']
    
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
    

    
    # Data paths
    if 'data' in config:
        if args.train_images is None and 'train_images' in config['data']:
            args.train_images = config['data']['train_images']
        if args.train_labels is None and 'train_labels' in config['data']:
            args.train_labels = config['data']['train_labels']
        if args.val_images is None and 'val_images' in config['data']:
            args.val_images = config['data']['val_images'] or None
        if args.val_labels is None and 'val_labels' in config['data']:
            args.val_labels = config['data']['val_labels'] or None
        if not hasattr(args, 'val_fraction') or args.val_fraction is None:
            args.val_fraction = config['data'].get('val_fraction', 0.2)
    
    # Output
    if 'output' in config:
        if args.output_dir is None and 'checkpoint_dir' in config['output']:
            args.output_dir = config['output']['checkpoint_dir']

    # Derive output_dir from experiment_name when not explicitly given
    if args.output_dir is None:
        experiment_name = config.get('experiment_name', '')
        if experiment_name:
            args.output_dir = os.path.join('./checkpoints', experiment_name)
    
    # Ensure parent checkpoints directory exists if using default path
    if args.output_dir and args.output_dir.startswith('./checkpoints'):
        os.makedirs('./checkpoints', exist_ok=True)
    
    # Seed
    if args.seed is None:
        args.seed = config.get('seed', 42)  # Default to 42 if not in config
    
    # Apply final defaults if still None
    if args.num_classes is None:
        args.num_classes = 5  # Default for task 1 (BG + 4 classes)
    if args.learning_rate is None:
        args.learning_rate = 1e-4
    if args.weight_decay is None:
        args.weight_decay = 1e-5
    if args.max_epochs is None:
        args.max_epochs = 100
    if args.batch_size is None:
        args.batch_size = 2
    if args.num_workers is None:
        args.num_workers = 4
    if args.val_interval is None:
        args.val_interval = 5
    if args.patience is None:
        args.patience = 20
    if args.output_dir is None:
        args.output_dir = './checkpoints'
        os.makedirs('./checkpoints', exist_ok=True)
    if args.compute_hd95 is None:
        args.compute_hd95 = True
    
    return args


def setup_logging(output_dir: str):
    """
    Set up logging to both file and console.
    
    Args:
        output_dir: Directory to save log file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load config if provided
    config = {}
    if args.config:
        config = load_config(args.config)
        args = merge_config_with_args(config, args)
    else:
        # Apply defaults when no config file is provided
        args.config_dict = {}
        if args.num_classes is None:
            args.num_classes = 5
        if args.learning_rate is None:
            args.learning_rate = 1e-4
        if args.weight_decay is None:
            args.weight_decay = 1e-5
        if args.max_epochs is None:
            args.max_epochs = 100
        if args.batch_size is None:
            args.batch_size = 2
        if args.num_workers is None:
            args.num_workers = 4
        if args.val_interval is None:
            args.val_interval = 5
        if args.patience is None:
            args.patience = 20
        if args.output_dir is None:
            args.output_dir = './checkpoints'
        if args.seed is None:
            args.seed = 42
        if args.val_fraction is None:
            args.val_fraction = 0.2
        if args.compute_hd95 is None:
            args.compute_hd95 = True
    
    # Set random seed for reproducibility
    monai.utils.set_determinism(args.seed)
    torch.manual_seed(args.seed)
    
    # Setup logging
    setup_logging(args.output_dir)
    
    # Log configuration
    logging.info("="*60)
    logging.info("Task 1: Liver Tumor Segmentation Training")
    logging.info("="*60)
    logging.info(f"Configuration:")
    for key, value in vars(args).items():
        logging.info(f"  {key}: {value}")
    logging.info("="*60)
    
    # Validate required arguments
    required_args = ['train_images', 'train_labels']
    missing_args = [arg for arg in required_args if not getattr(args, arg)]
    if missing_args:
        raise ValueError(f"Missing required arguments: {', '.join(missing_args)}")

    # Load data
    logging.info("Loading data...")
    val_fraction = getattr(args, 'val_fraction', 0.2)

    if args.val_images and args.val_labels:
        # Explicit validation directories supplied
        train_files, val_files = load_data(
            args.train_images,
            args.train_labels,
            args.val_images,
            args.val_labels,
        )
    else:
        # No val dirs -> random split from the training directory
        logging.info(
            f"No validation paths provided - performing random split "
            f"(val_fraction={val_fraction}, seed={args.seed})."
        )
        train_files, val_files = load_data_random(
            images_dir=args.train_images,
            labels_dir=args.train_labels,
            val_fraction=val_fraction,
            seed=args.seed,
        )

    if len(train_files) == 0:
        raise ValueError("No training data found!")
    if len(val_files) == 0:
        raise ValueError("No validation data found!")
    
    # Get data transforms
    train_transforms, val_transforms = get_data_transforms()
    
    # Create data loaders
    logging.info("Creating data loaders...")
    train_loader, val_loader = get_data_loaders(
        train_files,
        val_files,
        train_transforms,
        val_transforms,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Initialize model
    arch_type = config.get('architecture', {}).get('type', 'unet')
    logging.info(f"Initializing model: architecture='{arch_type}'")
    model = build_model(
        config=config,
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        class_names=args.class_names,
    )
    
    # Set up trainer
    logging.info("Setting up trainer...")
    if torch.cuda.is_available():
        device = "cuda"
        logging.info(f"Using device: cuda (GPU: {torch.cuda.get_device_name(0)})")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = "cpu"
        logging.warning("CUDA not available! Training on CPU will be very slow.")
        logging.info(f"Using device: {device}")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=args.output_dir,
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
    trainer.train(
        early_stopping_metric='val_loss',
        higher_is_better=False
    )

    # Save architecture metadata alongside the checkpoint so inferer.py and
    # evaluate.py can reconstruct the same model class without manual edits.
    arch_cfg = config.get('architecture', {})
    arch_type = arch_cfg.get('type', 'unet')
    model_cfg_data = {
        'arch_type':   arch_type,
        'arch_params': arch_cfg.get(arch_type, {}),
        'num_classes': args.num_classes,
    }
    model_cfg_path = Path(args.output_dir) / 'model_config.json'
    with open(model_cfg_path, 'w', encoding='utf-8') as f:
        json.dump(model_cfg_data, f, indent=2)
    logging.info(f"Architecture config saved to: {model_cfg_path}")
    
    # Save final metrics and generate plots
    logging.info("Saving metrics and generating plots...")
    metric_groups = {
        "Loss": ["train_loss", "val_loss"],
        "Dice Scores": [m for m in trainer.metric_tracker.metrics if "dice" in m.lower()],
        "HD95 Scores": [m for m in trainer.metric_tracker.metrics if "hd95" in m.lower()]
    }
    
    try:
        trainer.metric_tracker.plot_metrics(metric_groups)
        trainer.metric_tracker.save()
    except Exception as e:
        logging.error(f"Failed to save metrics and generate plots: {e}")
    
    # Log final summary
    summary = trainer.metric_tracker.get_summary()
    logging.info("="*60)
    logging.info("Training completed! Final metrics summary:")
    logging.info("="*60)
    for metric, stats in summary.items():
        logging.info(f"{metric}:")
        for stat_name, value in stats.items():
            logging.info(f"  {stat_name}: {value:.4f}")
    logging.info("="*60)
    
    logging.info(f"Best model saved to: {os.path.join(args.output_dir, 'best_model.pth')}")
    logging.info("Training complete!")


if __name__ == '__main__':
    main()
