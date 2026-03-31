"""
SegResNet segmentation model for Task 1.

MONAI SegResNet: residual encoder-decoder without skip connections.
Lighter than UNet; good for memory-constrained environments.
Reference: https://arxiv.org/abs/1810.11654

YAML config block (under architecture.segresnet):
  init_filters:   16              # base number of filters
  blocks_down:    [1, 2, 2, 4]   # residual blocks per encoder level
  blocks_up:      [1, 1, 1]      # residual blocks per decoder level
  dropout_prob:   0.0            # dropout probability (0 = disabled)
  roi_size:       [128, 128, 128]
  sw_batch_size:  4
"""
import torch.nn as nn
from monai.networks.nets import SegResNet

try:
    from src.base_model import SegmentationModelBase
except ModuleNotFoundError:
    from base_model import SegmentationModelBase


class SegmentationModel(SegmentationModelBase):
    """SegResNet-based 3-D segmentation model."""

    def _build_network(self, num_classes: int, arch_config: dict) -> nn.Module:
        return SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            init_filters=arch_config.get('init_filters', 16),
            blocks_down=arch_config.get('blocks_down', [1, 2, 2, 4]),
            blocks_up=arch_config.get('blocks_up', [1, 1, 1]),
            dropout_prob=arch_config.get('dropout_prob', 0.0),
        )
