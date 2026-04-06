"""
Segmentation Data Loader
Handles loading CT scans and segmentation masks for Task 1
"""
import os
import glob
import logging
from typing import List, Dict, Tuple, Optional

from sklearn.model_selection import train_test_split
from monai.data import Dataset, CacheDataset, DataLoader


def filter_images_with_labels(images: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
    """
    Filter images to keep only those that have corresponding labels.
    
    Args:
        images: List of image file paths
        labels: List of label file paths
        
    Returns:
        Tuple of (filtered_images, filtered_labels) with matching pairs
    """
    # Build a mapping from label file name (without extension) to label path
    label_map = {os.path.basename(lbl): lbl for lbl in labels}
    filtered_images = []
    filtered_labels = []
    
    for img_path in images:
        img_name = os.path.basename(img_path)
        if img_name not in label_map:
            logging.warning(f"No label found for image {img_name}")
            continue
        
        filtered_images.append(img_path)
        filtered_labels.append(label_map[img_name])
        
    return filtered_images, filtered_labels


def load_data(train_images_dir: str, train_labels_dir: str, 
              val_images_dir: str, val_labels_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Load training and validation data from directories.
    
    Args:
        train_images_dir: Directory containing training images
        train_labels_dir: Directory containing training labels
        val_images_dir: Directory containing validation images
        val_labels_dir: Directory containing validation labels
        
    Returns:
        Tuple of (train_files, val_files) where each is a list of dicts with 'image' and 'label' keys
    """
    # Load and filter training data
    train_images = sorted(glob.glob(os.path.join(train_images_dir, "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(train_labels_dir, "*.nii.gz")))
    train_images, train_labels = filter_images_with_labels(train_images, train_labels)
    
    # Load and filter validation data
    val_images = sorted(glob.glob(os.path.join(val_images_dir, "*.nii.gz")))
    val_labels = sorted(glob.glob(os.path.join(val_labels_dir, "*.nii.gz")))
    val_images, val_labels = filter_images_with_labels(val_images, val_labels)
    
    logging.info(f"Found {len(train_images)} training images and {len(val_images)} validation images")
    
    # Create data dictionaries
    train_files = [
        {"image": img_path, "label": lbl_path}
        for img_path, lbl_path in zip(train_images, train_labels)
    ]
    val_files = [
        {"image": img_path, "label": lbl_path}
        for img_path, lbl_path in zip(val_images, val_labels)
    ]
    
    return train_files, val_files


def load_data_random(
    images_dir: str,
    labels_dir: str,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load all images/labels from a single directory and perform a random
    train/val split.

    Args:
        images_dir:   Directory containing ``*.nii.gz`` image files.
        labels_dir:   Directory containing ``*.nii.gz`` label files.
        val_fraction: Fraction of samples to reserve for validation (0-1).
        seed:         Random seed for reproducibility.

    Returns:
        ``(train_files, val_files)`` - lists of ``{"image": …, "label": …}`` dicts.
    """
    all_images = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
    all_labels = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))
    all_images, all_labels = filter_images_with_labels(all_images, all_labels)

    all_files = [
        {"image": img, "label": lbl}
        for img, lbl in zip(all_images, all_labels)
    ]

    train_files, val_files = train_test_split(
        all_files, test_size=val_fraction, random_state=seed
    )

    logging.info(
        f"Random split -> {len(train_files)} train / {len(val_files)} val "
        f"(val_fraction={val_fraction}, seed={seed})"
    )

    return train_files, val_files


def get_data_loaders(train_files: List[Dict], val_files: List[Dict], 
                     train_transforms, val_transforms,
                     batch_size: int = 2, num_workers: int = 4,
                     cache_rate: float = 0.5) -> Tuple[DataLoader, DataLoader]:
    """
    Create MONAI data loaders from file lists with optional caching.
    
    Args:
        train_files: List of training file dictionaries
        val_files: List of validation file dictionaries
        train_transforms: Training data transforms
        val_transforms: Validation data transforms
        batch_size: Batch size for training
        num_workers: Number of worker processes
        cache_rate: Fraction of data to cache in memory (0-1)
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets with optional caching
    cache_rate = max(0.0, min(1.0, cache_rate))  # Clamp to [0, 1]
    
    if cache_rate > 0:
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=cache_rate)
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=cache_rate)
        logging.info(f"CacheDataset enabled: cache_rate={cache_rate}")
    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        val_ds = Dataset(data=val_files, transform=val_transforms)
        logging.info("Using standard Dataset (no caching)")
    
    # Determine if persistent workers should be used
    persistent = num_workers > 0
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=1,  # Always use batch size 1 for validation
        num_workers=num_workers,
        persistent_workers=persistent
    )
    
    return train_loader, val_loader
