"""
DenseNet-121 classification model for Task 2.

MONAI DenseNet-121: dense connectivity — every layer receives feature maps from
all preceding layers in its dense block.  Typically uses fewer parameters than
ResNet-50 (~8 M) while achieving comparable accuracy and is popular in medical
imaging classification tasks.
Reference: https://arxiv.org/abs/1608.06993

YAML config block (under model.densenet121):
  growth_rate:  32   # feature maps added per dense layer
  bn_size:       4   # bottleneck size multiplier (bn_size × growth_rate channels)
  dropout_prob: 0.0  # per-layer dropout probability (try 0.1–0.2 to combat overfitting)
"""
import torch.nn as nn
from monai.networks.nets import DenseNet121

try:
    from src.base_model import ClassificationModelBase
except ModuleNotFoundError:
    from base_model import ClassificationModelBase


class ClassificationModel(ClassificationModelBase):
    """MONAI DenseNet-121-based 3-D classification model."""

    def _build_network(self, num_classes: int, arch_config: dict) -> nn.Module:
        return DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            growth_rate=arch_config.get('growth_rate', 32),
            bn_size=arch_config.get('bn_size', 4),
            dropout_prob=arch_config.get('dropout_prob', 0.0),
        )
