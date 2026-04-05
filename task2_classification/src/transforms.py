"""
Data Transforms for Task 2 Classification
Defines augmentation and preprocessing transforms for tumor ROI classification.

Note: Input numpy arrays are assumed to have been preprocessed by prepare_dataset.py,
which applies abdomen HU windowing (center=40, width=400) giving values in [-160, 240].
These raw HU values are preserved (no normalization) to maintain full gradient signal
for MONAI models, which are optimized for raw medical imaging intensities.
"""
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    ScaleIntensityRange,
    RandRotate90,
    RandFlip,
    RandGaussianNoise,
    RandAffine,
    Resize,
    RandAdjustContrast,
    RandGaussianSmooth,
)

# Abdomen HU window used during preprocessing (must match prepare_dataset.py)
_HU_MIN = -160.0   # window_center(40) - window_width(400) / 2
_HU_MAX =  240.0   # window_center(40) + window_width(400) / 2


def get_train_transforms(spatial_size=None, enable_augmentation=True):
    """
    Get training transforms with optional augmentation.

    Expects pre-windowed numpy arrays with HU values in [_HU_MIN, _HU_MAX].

    Args:
        spatial_size: Target spatial size (z, y, x) for resizing. 
        enable_augmentation: If False, returns validation transforms (no augmentation).

    Returns:
        Composed transform pipeline.
    """
    if spatial_size is None:
        spatial_size = (96, 96, 96)
    
    base_transforms = [
        EnsureChannelFirst(channel_dim='no_channel'),
        ScaleIntensityRange(a_min=_HU_MIN, a_max=_HU_MAX, b_min=-1.0, b_max=1.0, clip=True),
        Resize(spatial_size=spatial_size),
    ]
    
    if not enable_augmentation:
        return Compose(base_transforms)
    
    # Add augmentation transforms
    augmentation_transforms = [
        RandRotate90(prob=0.5, spatial_axes=[0, 1]),
        RandFlip(prob=0.5, spatial_axis=0),
        RandFlip(prob=0.5, spatial_axis=1),
        RandFlip(prob=0.5, spatial_axis=2),
        RandAffine(
            prob=0.5,
            rotate_range=(0.17, 0.17, 0.17),  # ±10 degrees
            scale_range=(0.1, 0.1, 0.1),
            mode='bilinear',
            padding_mode='border'
        ),
        RandAdjustContrast(prob=0.5, gamma=(0.8, 1.2)),
        RandGaussianNoise(prob=0.3, mean=0.0, std=0.01),
        RandGaussianSmooth(prob=0.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0))
    ]
    
    return Compose(base_transforms + augmentation_transforms)


def get_val_transforms(spatial_size=None):
    """
    Get validation/test transforms without augmentation.

    Args:
        spatial_size: Target spatial size (z, y, x) for resizing.

    Returns:
        Composed transform pipeline.
    """
    if spatial_size is None:
        spatial_size = (96, 96, 96)
    # Use raw HU values for validation/test transforms as well
    return Compose([
        EnsureChannelFirst(channel_dim='no_channel'),
        ScaleIntensityRange(a_min=_HU_MIN, a_max=_HU_MAX, b_min=-1.0, b_max=1.0, clip=True),
        Resize(spatial_size=spatial_size),
    ])
