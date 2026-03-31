"""
Classification Metrics for Task 2
Computes accuracy, precision, recall, F1, and per-class metrics
"""
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import torch
import numpy as np
import logging


class ClassificationMetrics:
    """
    Metrics calculator for multi-class classification.
    """
    
    def __init__(self, num_classes=3):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes (default: 3)
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metric accumulators."""
        self.predictions = []
        self.probabilities = []
        self.targets = []
    
    def update(self, y_pred, y_true):
        """
        Update metric states with new predictions and targets.
        
        Args:
            y_pred: Model predictions (logits before softmax) [batch_size, num_classes]
            y_true: Ground truth labels [batch_size]
        """
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        
        # Store probabilities (after softmax)
        probs = self._softmax(y_pred)
        self.probabilities.extend(probs)
        
        # Convert probabilities to class predictions
        predictions = np.argmax(probs, axis=1)
        self.predictions.extend(predictions)
        self.targets.extend(y_true)
    
    def _softmax(self, x):
        """Apply softmax to logits."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def compute(self):
        """
        Compute all metrics.
        
        Returns:
            dict: Dictionary containing all computed metrics
        """
        predictions = np.array(self.predictions)
        probabilities = np.array(self.probabilities)
        targets = np.array(self.targets)
        
        if len(predictions) == 0:
            logging.warning("No predictions to compute metrics from")
            return self._empty_metrics()
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, average='macro', zero_division=0),
            'recall': recall_score(targets, predictions, average='macro', zero_division=0),
            'f1': f1_score(targets, predictions, average='macro', zero_division=0),
        }
        
        # Per-class metrics
        per_class_precision = precision_score(targets, predictions, average=None, zero_division=0)
        per_class_recall = recall_score(targets, predictions, average=None, zero_division=0)
        per_class_f1 = f1_score(targets, predictions, average=None, zero_division=0)
        
        for i in range(self.num_classes):
            if i < len(per_class_precision):
                metrics[f'class_{i}_precision'] = per_class_precision[i]
                metrics[f'class_{i}_recall'] = per_class_recall[i]
                metrics[f'class_{i}_f1'] = per_class_f1[i]
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        metrics['confusion_matrix'] = cm
        
        return metrics
    
    def _empty_metrics(self):
        """Return empty metrics dictionary."""
        metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
        }
        for i in range(self.num_classes):
            metrics[f'class_{i}_precision'] = 0.0
            metrics[f'class_{i}_recall'] = 0.0
            metrics[f'class_{i}_f1'] = 0.0
        metrics['confusion_matrix'] = np.zeros((self.num_classes, self.num_classes))
        return metrics
    
    def get_classification_report(self, class_names=None):
        """
        Get sklearn classification report.
        
        Args:
            class_names: Optional list of class names
            
        Returns:
            str: Classification report string
        """
        if len(self.predictions) == 0:
            return "No predictions available"
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        return classification_report(
            targets,
            predictions,
            target_names=class_names,
            zero_division=0
        )
