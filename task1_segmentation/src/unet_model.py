"""
UNet segmentation model for Task 1.

MONAI UNet: encoder-decoder with skip connections and residual units.
Good general-purpose baseline for 3D medical image segmentation.

YAML config block (under architecture.unet):
  channels:       [16, 32, 64, 128, 256]   # feature maps per level
  strides:        [2, 2, 2, 2]             # downsampling strides (len = len(channels)-1)
  num_res_units:  2                         # residual units per block
  norm:           "BATCH"                  # BATCH | INSTANCE | GROUP
  roi_size:       [160, 160, 160]          # sliding-window patch size
  sw_batch_size:  4                        # patches per sliding-window forward pass
"""
import torch.nn as nn
from monai.networks.nets import UNet
from monai.networks.layers import Norm

try:
    from src.base_model import SegmentationModelBase
except ModuleNotFoundError:
    from base_model import SegmentationModelBase

_NORM_MAP = {
    'BATCH':    Norm.BATCH,
    'INSTANCE': Norm.INSTANCE,
    'GROUP':    Norm.GROUP,
}


class SegmentationModel(SegmentationModelBase):
    """UNet-based 3-D segmentation model."""

    def _build_network(self, num_classes: int, arch_config: dict) -> nn.Module:
        norm_key = arch_config.get('norm', 'BATCH').upper()
        norm = _NORM_MAP.get(norm_key, Norm.BATCH)

        return UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            channels=tuple(arch_config.get('channels', [16, 32, 64, 128, 256])),
            strides=tuple(arch_config.get('strides', [2, 2, 2, 2])),
            num_res_units=arch_config.get('num_res_units', 2),
            norm=norm,
        )
