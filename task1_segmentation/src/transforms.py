import warnings
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandAdjustContrastd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd
)
from monai.transforms.post.dictionary import Invertd

# Suppress harmless MONAI warning about class balanced sampling when no foreground samples found.
# This occurs with augmentation enabled due to RandAffined randomly cropping label regions.
# MONAI handles this gracefully by setting pos_ratio=0, so training proceeds normally.
warnings.filterwarnings("ignore", message=".*Num foregrounds 0.*")


def get_data_transforms(enable_augmentation: bool = False):
    """
    Get training and validation transforms.
    
    Args:
        enable_augmentation: If True, apply spatial and intensity augmentations to training data.
                           Validation data is not augmented regardless of this setting.
    
    Returns:
        Tuple of (train_transforms, val_transforms)
    """
    # Base transforms (common to all training samples)
    base_train_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
    ]
    
    # Augmentation transforms (applied only if enabled)
    augmentation_transforms = []
    if enable_augmentation:
        augmentation_transforms = [
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),  # LR flip
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),  # AP flip
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),  # SI flip
            RandAffined(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=0.8,
                spatial_size=(96, 96, 96),
                rotate_range=(0.26, 0.26, 0.26),  # ~15 degrees in radians
                scale_range=(0.1, 0.1, 0.1),
                padding_mode="border",
            ),
            RandGaussianNoised(keys=["image"], prob=0.5, std=0.01),
            RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.9, 1.1)),
        ]
    
    train_transforms = Compose(
        base_train_transforms +
        augmentation_transforms +
        [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
        ]
    )
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
    ])
    
    return train_transforms, val_transforms


def get_test_transforms(allow_missing_keys=False):
    """Test transforms without ToTensord to preserve metadata for Invertd."""
    return Compose([
        LoadImaged(keys=["image", "label"], allow_missing_keys=allow_missing_keys),
        EnsureChannelFirstd(keys=["image", "label"], allow_missing_keys=allow_missing_keys),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None, allow_missing_keys=allow_missing_keys),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest"), allow_missing_keys=allow_missing_keys),
        ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0, b_max=1, clip=True),
    ])


def get_post_transforms(test_transforms):
    """Inverse transforms to restore predictions to original image space."""
    return Compose([
        Invertd(
            keys="pred",
            transform=test_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True,
            to_tensor=False,
        )
    ])
