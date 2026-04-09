"""
ResNet-18 classification model for Task 2.

MONAI ResNet-18: lightweight 3-D residual network (~11 M parameters).
Good default for small datasets (< 500 ROIs) and quick baselines.
Reference: https://arxiv.org/abs/1512.03385

YAML config block (under model.resnet18):
  conv1_t_size:   7   # depth-axis kernel size for the first conv layer
  conv1_t_stride: 1   # use 1 for 64^3 volumes to avoid over-striding
"""
import torch.nn as nn
from monai.networks.nets import resnet18

try:
    from src.base_model import ClassificationModelBase
except ModuleNotFoundError:
    from base_model import ClassificationModelBase


class ClassificationModel(ClassificationModelBase):
    """MONAI ResNet-18-based 3-D classification model."""

    def _build_network(self, num_classes: int, arch_config: dict) -> nn.Module:
        return resnet18(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=num_classes,
            conv1_t_size=arch_config.get('conv1_t_size', 7),
            conv1_t_stride=arch_config.get('conv1_t_stride', 1),
        )
