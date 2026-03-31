"""
ResNet-50 classification model for Task 2.

MONAI ResNet-50: deeper residual network with bottleneck blocks (~46 M parameters).
Higher capacity than ResNet-18; consider when training data is > 500 ROIs or
when ResNet-18 under-fits.
Reference: https://arxiv.org/abs/1512.03385

YAML config block (under model.resnet50):
  conv1_t_size:   7   # depth-axis kernel size for the first conv layer
  conv1_t_stride: 2   # stride 2 downsamples the depth axis faster (saves memory)
"""
import torch.nn as nn
from monai.networks.nets import resnet50

try:
    from src.base_model import ClassificationModelBase
except ModuleNotFoundError:
    from base_model import ClassificationModelBase


class ClassificationModel(ClassificationModelBase):
    """MONAI ResNet-50-based 3-D classification model."""

    def _build_network(self, num_classes: int, arch_config: dict) -> nn.Module:
        return resnet50(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=num_classes,
            conv1_t_size=arch_config.get('conv1_t_size', 7),
            conv1_t_stride=arch_config.get('conv1_t_stride', 2),
            feed_forward=False,
            bias_downsample=True,
        )
