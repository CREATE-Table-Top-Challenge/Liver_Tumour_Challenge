"""
Task 2: Classification Model — base class and factory.

Architecture-specific classes live in their own files:
  src/resnet18_model.py    — MONAI ResNet-18
  src/resnet50_model.py    — MONAI ResNet-50
  src/densenet121_model.py — MONAI DenseNet-121

To add a new architecture: create <arch>_model.py with a ClassificationModel
subclass, override _build_network(), and register it in _ARCH_REGISTRY below.
"""
import importlib
import torch.nn as nn
from typing import Dict, Optional


class ClassificationModelBase(nn.Module):
    """
    Base class for all Task 2 classification models.

    Subclasses only need to implement _build_network(), which must return
    an nn.Module accepting (B, 1, D, H, W) and outputting logits (B, num_classes).
    All training / validation logic is handled by Trainer; this class holds
    the network and provides a uniform interface for the factory.
    """

    def __init__(self, num_classes: int = 5, arch_config: Optional[dict] = None):
        super().__init__()
        self.num_classes = num_classes
        self.net = self._build_network(num_classes, arch_config or {})

    def _build_network(self, num_classes: int, arch_config: dict) -> nn.Module:
        """Build and return the backbone network. Implemented by each subclass."""
        raise NotImplementedError

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

# Maps model_type name -> fully-qualified module path.
# The fallback in build_model() handles the Docker/src-only context where
# sys.path points directly to /app/src (no 'src.' prefix).
_ARCH_REGISTRY: Dict[str, str] = {
    'resnet18':    'src.resnet18_model',
    'resnet50':    'src.resnet50_model',
    'densenet121': 'src.densenet121_model',
}


def build_model(
    config: dict,
    num_classes: int,
    model_type: Optional[str] = None,
) -> ClassificationModelBase:
    """
    Instantiate the classification model specified by config['model']['model_type'].

    Args:
        config:      Full YAML config dict containing a 'model' key.
        num_classes: Number of output classes.
        model_type:  Override for config['model']['model_type'] (e.g. from CLI).

    Returns:
        Initialised ClassificationModel subclass ready for training.

    Raises:
        ValueError: If the requested architecture is not registered.
    """
    model_cfg = config.get('model', {})
    arch_type = (model_type or model_cfg.get('model_type', 'resnet18')).lower()
    arch_params = model_cfg.get(arch_type, {})

    if arch_type not in _ARCH_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{arch_type}'. "
            f"Supported options: {', '.join(_ARCH_REGISTRY)}"
        )

    module_path = _ARCH_REGISTRY[arch_type]
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        # Docker context: sys.path points directly to /app/src, no 'src.' prefix
        module = importlib.import_module(module_path.split('.')[-1])

    ClassificationModel = getattr(module, 'ClassificationModel')
    return ClassificationModel(num_classes=num_classes, arch_config=arch_params)
