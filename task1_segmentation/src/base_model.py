"""
Task 1: Segmentation Model — base class and factory.

Architecture-specific classes live in their own files:
  src/unet_model.py       — MONAI UNet
  src/segresnet_model.py  — MONAI SegResNet
  src/swinunetr_model.py  — MONAI SwinUNETR

To add a new architecture: create <arch>_model.py with a SegmentationModel
subclass, override _build_network(), and register it in _ARCH_REGISTRY below.
"""
import importlib
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete, Compose
from typing import Dict, Any, Tuple, Optional, List


class SegmentationModelBase(nn.Module):
    """
    Base class for all Task 1 segmentation models.

    Subclasses only need to implement _build_network(), which must return
    an nn.Module accepting (B, 1, H, W, D) and outputting (B, C, H, W, D).
    All training/validation logic, loss, metrics, and optimisers are shared here.
    """

    def __init__(
        self,
        num_classes: int = 3,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        class_names: Optional[List[str]] = None,
        arch_config: Optional[dict] = None,
        compute_hd95: bool = True,
    ):
        """
        Args:
            num_classes:    Total output channels including background.
            learning_rate:  Adam optimiser learning rate.
            weight_decay:   Adam optimiser weight decay (L2 regularisation).
            class_names:    Foreground class names for metric logging.
                            Length must equal num_classes - 1.
            arch_config:    Architecture-specific sub-dict from the YAML
                            (e.g. config['architecture']['unet']).
                            May contain 'roi_size' and 'sw_batch_size'.
            compute_hd95:   Whether to compute HD95 metric during validation.
                            Set to False for faster training iterations.
        """
        super().__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.compute_hd95 = compute_hd95
        arch_config = arch_config or {}

        # Sliding-window parameters — overrideable per architecture in YAML
        self.roi_size = tuple(arch_config.get('roi_size', [160, 160, 160]))
        self.sw_batch_size = int(arch_config.get('sw_batch_size', 4))

        # Network — constructed by subclass
        self.net = self._build_network(num_classes, arch_config)

        # Shared loss and metrics
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.val_dice = DiceMetric(include_background=False, reduction="none")
        # Conditionally initialize HD95 metric — can be disabled for faster training
        self.val_hd95 = (
            HausdorffDistanceMetric(include_background=False, reduction="none", percentile=95)
            if compute_hd95
            else None
        )

        self.post_pred = Compose([AsDiscrete(argmax=True, to_onehot=num_classes)])
        self.post_label = Compose([AsDiscrete(to_onehot=num_classes)])

        if class_names is None:
            self.class_names = [f"class_{i}" for i in range(1, num_classes)]
        else:
            self.class_names = class_names

        self.best_metric = -1
        self.best_metric_epoch = -1

    def _build_network(self, num_classes: int, arch_config: dict) -> nn.Module:
        """Build and return the backbone network. Implemented by each subclass."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        images, labels = batch["image"], batch["label"]
        outputs = self(images)
        loss = self.loss_function(outputs, labels)
        return loss, {"train_loss": loss.item()}

    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        images, labels = batch["image"], batch["label"]
        val_outputs = sliding_window_inference(
            images, self.roi_size, self.sw_batch_size, self.net
        )
        loss = self.loss_function(val_outputs, labels)
        val_outputs = [self.post_pred(i) for i in decollate_batch(val_outputs)]
        val_labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.val_dice(y_pred=val_outputs, y=val_labels)
        if self.compute_hd95:
            self.val_hd95(y_pred=val_outputs, y=val_labels)
        return {"val_loss": loss.item()}

    def on_validation_epoch_end(self) -> Dict[str, Any]:
        dice_scores = self.val_dice.aggregate()
        if dice_scores.ndim > 1:
            dice_scores = dice_scores.mean(dim=0)
        metrics_dict = {}
        for i, class_name in enumerate(self.class_names):
            metrics_dict[f"val_dice_{class_name}"] = dice_scores[i].item()
        self.val_dice.reset()
        
        # Only compute HD95 metrics if enabled
        if self.compute_hd95:
            hd95_scores = self.val_hd95.aggregate()
            if hd95_scores.ndim > 1:
                # HD95 can be NaN when a class is absent in a volume; ignore those
                hd95_scores = torch.nanmean(hd95_scores, dim=0)
            for i, class_name in enumerate(self.class_names):
                metrics_dict[f"val_hd95_{class_name}"] = hd95_scores[i].item()
            self.val_hd95.reset()
        
        return metrics_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return optimizer, scheduler

    def get_progress_bar_dict(self) -> Dict[str, Any]:
        return {
            "lr": self.learning_rate,
            "best_metric": self.best_metric,
            "best_epoch": self.best_metric_epoch,
        }


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

# Maps architecture name -> fully-qualified module path.
# The fallback in build_model() handles the case where the arch files are
# run directly from the src/ directory (no 'src.' prefix on sys.path).
_ARCH_REGISTRY: Dict[str, str] = {
    'unet':      'src.unet_model',
    'segresnet': 'src.segresnet_model',
    'swinunetr': 'src.swinunetr_model',
}


def build_model(
    config: dict,
    num_classes: int,
    learning_rate: float,
    weight_decay: float = 1e-5,
    class_names: Optional[List[str]] = None,
) -> SegmentationModelBase:
    """
    Instantiate the segmentation model specified by config['architecture']['type'].

    Args:
        config:        Full YAML config dict containing an 'architecture' key.
        num_classes:   Number of output channels including background.
        learning_rate: Optimiser learning rate.
        weight_decay:  Optimiser weight decay (L2 regularisation).
        class_names:   Foreground class names (length = num_classes - 1).

    Returns:
        Initialised SegmentationModel subclass ready for training.

    Raises:
        ValueError: If the requested architecture is not registered.
    """
    arch_cfg = config.get('architecture', {})
    arch_type = arch_cfg.get('type', 'unet').lower()
    arch_params = arch_cfg.get(arch_type, {})
    
    # Extract compute_hd95 flag from model config (default True for backward compatibility)
    model_cfg = config.get('model', {})
    compute_hd95 = model_cfg.get('compute_hd95', True)

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

    SegmentationModel = getattr(module, 'SegmentationModel')
    return SegmentationModel(
        num_classes=num_classes,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        class_names=class_names,
        arch_config=arch_params,
        compute_hd95=compute_hd95,
    )
