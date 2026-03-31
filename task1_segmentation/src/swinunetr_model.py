"""
SwinUNETR segmentation model for Task 1.

MONAI SwinUNETR: Swin Transformer encoder + CNN decoder.
State-of-the-art accuracy but requires more GPU memory than UNet/SegResNet.
Reference: https://arxiv.org/abs/2201.01266

Note: MONAI >= 1.3 removed the img_size constructor argument — patch embedding
is now fully dynamic. roi_size is only used for sliding-window inference.

YAML config block (under architecture.swinunetr):
  feature_size:    48     # embedding dimension (must be divisible by 12)
  drop_rate:       0.0    # dropout rate
  attn_drop_rate:  0.0    # attention dropout rate
  use_checkpoint:  false  # gradient checkpointing (saves memory, slower training)
  roi_size:        [96, 96, 96]   # sliding-window patch size
  sw_batch_size:   2              # lower than UNet due to higher memory cost
"""
import torch.nn as nn
from monai.networks.nets import SwinUNETR

try:
    from src.base_model import SegmentationModelBase
except ModuleNotFoundError:
    from base_model import SegmentationModelBase


class SegmentationModel(SegmentationModelBase):
    """SwinUNETR-based 3-D segmentation model."""

    def _build_network(self, num_classes: int, arch_config: dict) -> nn.Module:
        return SwinUNETR(
            in_channels=1,
            out_channels=num_classes,
            feature_size=arch_config.get('feature_size', 48),
            drop_rate=arch_config.get('drop_rate', 0.0),
            attn_drop_rate=arch_config.get('attn_drop_rate', 0.0),
            use_checkpoint=arch_config.get('use_checkpoint', False),
        )
