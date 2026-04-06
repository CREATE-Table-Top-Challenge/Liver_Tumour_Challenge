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
        optimizer_config: Optional[dict] = None,
        scheduler_config: Optional[dict] = None,
        compute_hd95: bool = True,
    ):
        """
        Args:
            num_classes:    Total output channels including background.
            learning_rate:  Optimiser learning rate.
            weight_decay:   Optimiser weight decay (L2 regularisation).
            class_names:    Foreground class names for metric logging.
                            Length must equal num_classes - 1.
            arch_config:    Architecture-specific sub-dict from the YAML
                            (e.g. config['architecture']['unet']).
                            May contain 'roi_size' and 'sw_batch_size'.
            optimizer_config: Optimizer configuration dict with keys:
                            'type' (adam|sgd|adamw), 'weight_decay', 'momentum', 'betas'
            scheduler_config: Scheduler configuration dict with keys:
                            'type' (cosine|step|reduce_on_plateau|none), 'T_max', 'eta_min', etc.
            compute_hd95:   Whether to compute HD95 metric during validation.
                            Set to False for faster training iterations.
        """
        super().__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_config = optimizer_config or {}
        self.scheduler_config = scheduler_config or {}
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
        """Configure optimizer and scheduler based on config settings."""
        # Optimizer configuration (convert YAML string values to float)
        opt_type = self.optimizer_config.get('type', 'adam').lower()
        lr = float(self.optimizer_config.get('lr', self.learning_rate))
        weight_decay = float(self.optimizer_config.get('weight_decay', self.weight_decay))
        eps = float(self.optimizer_config.get('eps', 1e-8))
        
        if opt_type == 'sgd':
            momentum = float(self.optimizer_config.get('momentum', 0.9))
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif opt_type == 'adamw':
            betas_list = self.optimizer_config.get('betas', [0.9, 0.999])
            betas = tuple(float(b) for b in betas_list)
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
                eps=eps
            )
        else:  # Default to Adam
            betas_list = self.optimizer_config.get('betas', [0.9, 0.999])
            betas = tuple(float(b) for b in betas_list)
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
                eps=eps
            )
        
        # Scheduler configuration (convert YAML string values to float/int)
        scheduler_type = self.scheduler_config.get('type', 'reduce_on_plateau').lower()
        scheduler = None
        
        if scheduler_type == 'cosine':
            T_max = int(self.scheduler_config.get('T_max', 100))
            eta_min = float(self.scheduler_config.get('eta_min', 1e-6))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=eta_min
            )
        elif scheduler_type == 'step':
            step_size = int(self.scheduler_config.get('step_size', 30))
            gamma = float(self.scheduler_config.get('gamma', 0.1))
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif scheduler_type == 'reduce_on_plateau':
            patience = int(self.scheduler_config.get('patience', 5))
            factor = float(self.scheduler_config.get('factor', 0.5))
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=factor,
                patience=patience
            )
        # elif scheduler_type == 'none': scheduler remains None
        
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
        config:        Full YAML config dict containing 'architecture', 'optimizer', and 'scheduler' keys.
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
    
    # Extract compute_hd95 flag from training config (default True for backward compatibility)
    training_cfg = config.get('training', {})
    compute_hd95 = training_cfg.get('compute_hd95', True)
    
    # Extract optimizer and scheduler configs
    optimizer_cfg = config.get('optimizer', {})
    scheduler_cfg = config.get('scheduler', {})

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
        optimizer_config=optimizer_cfg,
        scheduler_config=scheduler_cfg,
        compute_hd95=compute_hd95,
    )
