"""
Classification Data Loader
Handles loading tumor ROI crops and labels for Task 2
"""
import os
import glob
import logging
import random
import numpy as np
import torch
from typing import List, Dict, Tuple
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold


class ROIDataset(Dataset):
    """
    Dataset for loading pre-cropped tumor ROIs.
    Expects directory structure: data_dir/class_name/*.npy
    """
    
    def __init__(self, data_dir: str, transforms=None):
        """
        Initialize ROI dataset.
        
        Args:
            data_dir: Root directory with class subdirectories
            transforms: Optional transforms to apply
        """
        self.data_files = []
        self.labels = []
        self.transforms = transforms
        self.class_names = []
        
        # Load from directory structure: data_dir/class_name/*.npy
        for class_idx, class_name in enumerate(sorted(os.listdir(data_dir))):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                self.class_names.append(class_name)
                file_count = 0
                for file_name in os.listdir(class_dir):
                    if file_name.endswith('.npy') or file_name.endswith('.nii.gz'):
                        self.data_files.append(os.path.join(class_dir, file_name))
                        self.labels.append(class_idx)
                        file_count += 1
                logging.info(f"Loaded {file_count} samples for class {class_idx} ({class_name})")
        
        logging.info(f"Total samples: {len(self.data_files)}")
        logging.info(f"Classes: {self.class_names}")
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        # Load ROI
        file_path = self.data_files[idx]
        
        if file_path.endswith('.npy'):
            data = np.load(file_path)
        elif file_path.endswith('.nii.gz') or file_path.endswith('.nii'):
            import nibabel as nib
            data = nib.load(file_path).get_fdata()
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Ensure numpy array
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # Convert to float32
        data = data.astype(np.float32)
        
        # Apply transforms (transforms will handle channel dimension)
        if self.transforms:
            data = self.transforms(data)
        else:
            # Return raw numpy array — callers that need a tensor should supply transforms
            data = data   # shape: (D, H, W), float32
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {"image": data, "label": label, "file_path": file_path}


def load_data(data_dir: str, transforms=None) -> ROIDataset:
    """
    Load ROI dataset from directory.
    
    Args:
        data_dir: Root directory containing class subdirectories
        transforms: Optional data transforms
        
    Returns:
        ROIDataset instance
    """
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
    
    dataset = ROIDataset(data_dir, transforms=transforms)
    
    if len(dataset) == 0:
        raise ValueError(f"No data found in {data_dir}")
    
    return dataset


def get_data_loaders(train_dataset: Dataset, val_dataset: Dataset = None,
                     batch_size: int = 16, num_workers: int = 4,
                     shuffle_train: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and validation.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        batch_size: Batch size for data loading
        num_workers: Number of worker processes
        shuffle_train: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, val_loader) or (train_loader, None)
    """
    # Determine if persistent workers should be used
    persistent = num_workers > 0
    
    # Create training loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        persistent_workers=persistent,
        pin_memory=True
    )
    
    # Create validation loader if dataset provided
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent,
            pin_memory=True
        )
    
    return train_loader, val_loader


def split_dataset(dataset: Dataset, train_ratio: float = 0.8,
                  random_seed: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Randomly split dataset into training and validation sets.

    Args:
        dataset: Dataset to split.
        train_ratio: Fraction of data for training (0-1).
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_subset, val_subset) — both are ``Subset`` objects
        backed by the original dataset.
    """
    rng = random.Random(random_seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    n_train = max(1, int(len(indices) * train_ratio))
    train_indices = indices[:n_train]
    val_indices   = indices[n_train:]

    logging.info(f"Random split: {len(train_indices)} train, {len(val_indices)} val")

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def create_cv_folds(
    dataset: Dataset,
    n_splits: int = 5,
    random_seed: int = 42,
) -> List[Tuple[Subset, Subset]]:
    """
    Create random k-fold cross-validation splits.

    Uses ``KFold`` for a random split
    distribution.  Returns a list of ``(train_subset, val_subset)`` pairs
    — one per fold — both backed by the original dataset so transforms
    can be applied externally (via ``TransformSubset``).

    Args:
        dataset:     The dataset to split.
        n_splits:    Number of folds (default 5).
        random_seed: Controls fold shuffling for reproducibility.

    Returns:
        List of ``(train_subset, val_subset)`` tuples, length ``n_splits``.
    """
    indices = np.arange(len(dataset))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    folds: List[Tuple[Subset, Subset]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
        train_subset = Subset(dataset, train_idx.tolist())
        val_subset   = Subset(dataset, val_idx.tolist())
        logging.info(f"Fold {fold_idx + 1}/{n_splits}: "
                     f"{len(train_idx)} train, {len(val_idx)} val")
        folds.append((train_subset, val_subset))

    return folds
