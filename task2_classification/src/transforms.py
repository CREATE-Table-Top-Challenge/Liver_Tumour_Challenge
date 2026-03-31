"""
Data Transforms for Task 2 Classification
Defines augmentation and preprocessing transforms for tumor ROI classification.

Note: Input numpy arrays are assumed to have been preprocessed by prepare_dataset.py,
which applies abdomen HU windowing (center=40, width=400) giving values in [-160, 240].
ScaleIntensityRange normalises this fixed range to [0, 1] deterministically.
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


def get_train_transforms(spatial_size=(64, 64, 64)):
    """
    Get training transforms with augmentation.

    Expects pre-windowed numpy arrays with HU values in [_HU_MIN, _HU_MAX].

    Args:
        spatial_size: Target spatial size for resizing.

    Returns:
        Composed transform pipeline.
    """
    return Compose([
        EnsureChannelFirst(channel_dim='no_channel'),
        ScaleIntensityRange(a_min=_HU_MIN, a_max=_HU_MAX, b_min=0.0, b_max=1.0, clip=True),
        Resize(spatial_size=spatial_size),
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
    ])


def get_val_transforms(spatial_size=(64, 64, 64)):
    """
    Get validation/test transforms without augmentation.

    Expects pre-windowed numpy arrays with HU values in [_HU_MIN, _HU_MAX].

    Args:
        spatial_size: Target spatial size for resizing.

    Returns:
        Composed transform pipeline.
    """
    return Compose([
        EnsureChannelFirst(channel_dim='no_channel'),
        ScaleIntensityRange(a_min=_HU_MIN, a_max=_HU_MAX, b_min=0.0, b_max=1.0, clip=True),
        Resize(spatial_size=spatial_size),
    ])
