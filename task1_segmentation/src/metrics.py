import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric

class SegmentationMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95)
        self.reset()

    def reset(self):
        self.dice_scores = []
        self.hd95_scores = []

    def update(self, preds, labels):
        # preds, labels: (B, C, ...)
        self.dice_scores.append(self.dice_metric(preds, labels))
        self.hd95_scores.append(self.hd95_metric(preds, labels))

    def compute(self):
        dice = torch.cat(self.dice_scores, dim=0).mean(dim=0).cpu().numpy()
        hd95 = torch.cat(self.hd95_scores, dim=0).mean(dim=0).cpu().numpy()
        return dice, hd95
