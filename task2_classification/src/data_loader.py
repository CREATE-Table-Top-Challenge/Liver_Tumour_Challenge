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
import pandas as pd
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit, ShuffleSplit


class NIfTIDataset(Dataset):
    """
    Dataset for loading tumor ROIs from NIfTI files with CSV labels mapping.
    Expects flat directory structure: data_dir/*.nii.gz with accompanying labels.csv
    """
    
    def __init__(self, data_dir: str, labels_csv: str, transforms=None):
        """
        Initialize dataset from NIfTI files and CSV labels.
        
        Args:
            data_dir: Directory containing NIfTI ROI files
            labels_csv: Path to CSV file with columns: patient_id, type
            transforms: Optional transforms to apply
        """
        self.data_dir = data_dir
        self.transforms = transforms
        self.data_files = []
        self.labels = []
        self.class_names = []
        self.patient_ids = []
        
        if not os.path.isfile(labels_csv):
            raise ValueError(f"Labels CSV not found: {labels_csv}")
        
        df = pd.read_csv(labels_csv)
        if not all(col in df.columns for col in ['patient_id', 'type']):
            raise ValueError("CSV must contain 'patient_id' and 'type' columns")
        
        # Create mapping from class name to label index
        unique_classes = sorted(df['type'].unique())
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}
        self.class_names = unique_classes
        
        # Load data files and labels from CSV
        for _, row in df.iterrows():
            patient_id = str(row['patient_id'])
            class_name = str(row['type'])
            
            # Find matching ROI file
            roi_file = None
            for ext in ['.nii.gz', '.nii']:
                candidate = os.path.join(data_dir, f"{patient_id}{ext}")
                if os.path.isfile(candidate):
                    roi_file = candidate
                    break
            
            if roi_file is None:
                logging.warning(f"ROI file not found for patient {patient_id} - skipping")
                continue
            
            if class_name not in class_to_idx:
                logging.warning(f"Unknown class '{class_name}' for patient {patient_id} - skipping")
                continue
            
            self.data_files.append(roi_file)
            self.labels.append(class_to_idx[class_name])
            self.patient_ids.append(patient_id)
        
        logging.info(f"Loaded {len(self.data_files)} samples from {data_dir}")
        logging.info(f"Classes: {self.class_names}")
        logging.info(f"Class distribution: {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        file_path = self.data_files[idx]
        
        import nibabel as nib
        data = nib.load(file_path).get_fdata()
        
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        data = data.astype(np.float32)
        
        if self.transforms:
            data = self.transforms(data)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {"image": data, "label": label, "file_path": file_path}


def load_data(data_dir: str, labels_csv: str, transforms=None) -> NIfTIDataset:
    """
    Load ROI dataset from directory with CSV labels.
    
    Args:
        data_dir: Directory containing NIfTI ROI files
        labels_csv: Path to CSV file with patient_id and type columns
        transforms: Optional data transforms
        
    Returns:
        NIfTIDataset instance
    """
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
    
    if not os.path.isfile(labels_csv):
        raise ValueError(f"Labels CSV not found: {labels_csv}")
    
    dataset = NIfTIDataset(data_dir, labels_csv, transforms=transforms)
    
    if len(dataset) == 0:
        raise ValueError(f"No valid data found in {data_dir} matching CSV labels")
    
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
                  random_seed: int = 42, enable_stratified_split: bool = True) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into training and validation sets.
    Optionally maintains class distribution in both sets (stratified split).

    Args:
        dataset: Dataset to split.
        train_ratio: Fraction of data for training (0-1).
        random_seed: Random seed for reproducibility.
        enable_stratified_split: If True, use stratified split to maintain class distribution.
                                 If False, use random split.

    Returns:
        Tuple of (train_subset, val_subset) — both are ``Subset`` objects
        backed by the original dataset.
    """
    # Get labels for stratification
    if hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        raise ValueError("Dataset must have 'labels' attribute for split")
    
    test_ratio = 1.0 - train_ratio
    
    if enable_stratified_split:
        # Use StratifiedShuffleSplit to maintain class distribution
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed)
        train_indices, val_indices = list(splitter.split(np.arange(len(dataset)), labels))[0]
        split_type = "Stratified"
    else:
        # Use random split
        splitter = ShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed)
        train_indices, val_indices = list(splitter.split(np.arange(len(dataset))))[0]
        split_type = "Random"
    
    # Log class distribution
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    
    logging.info(f"{split_type} split: {len(train_indices)} train, {len(val_indices)} val")
    logging.info(f"  Train class distribution: {np.bincount(train_labels)}")
    logging.info(f"  Val class distribution: {np.bincount(val_labels)}")

    return Subset(dataset, train_indices.tolist()), Subset(dataset, val_indices.tolist())


def create_cv_folds(
    dataset: Dataset,
    n_splits: int = 5,
    random_seed: int = 42,
    enable_stratified_split: bool = True,
) -> List[Tuple[Subset, Subset]]:
    """
    Create k-fold cross-validation splits.

    Optionally uses ``StratifiedKFold`` to maintain class distribution in each fold,
    or ``KFold`` for random splits. Returns a list of ``(train_subset, val_subset)``
    pairs — one per fold — both backed by the original dataset so transforms can be
    applied externally (via ``TransformSubset``).

    Args:
        dataset:     The dataset to split.
        n_splits:    Number of folds (default 5).
        random_seed: Controls fold shuffling for reproducibility.
        enable_stratified_split: If True, use stratified k-fold to maintain class distribution.
                                 If False, use random k-fold.

    Returns:
        List of ``(train_subset, val_subset)`` tuples, length ``n_splits``.
    """
    # Get labels for stratification (if needed)
    if hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        raise ValueError("Dataset must have 'labels' attribute for k-fold")
    
    indices = np.arange(len(dataset))
    
    if enable_stratified_split:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        split_type = "Stratified k-fold"
        splits = splitter.split(indices, labels)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        split_type = "Random k-fold"
        splits = splitter.split(indices)

    folds: List[Tuple[Subset, Subset]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_subset = Subset(dataset, train_idx.tolist())
        val_subset   = Subset(dataset, val_idx.tolist())
        
        # Log class distribution for this fold
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        logging.info(f"{split_type} Fold {fold_idx + 1}/{n_splits}: "
                     f"{len(train_idx)} train, {len(val_idx)} val")
        logging.info(f"  Train class distribution: {np.bincount(train_labels)}")
        logging.info(f"  Val class distribution: {np.bincount(val_labels)}")
        
        folds.append((train_subset, val_subset))

    return folds
