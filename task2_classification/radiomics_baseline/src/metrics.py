"""
Classification metrics for the radiomics baseline.

compute_metrics() returns a plain dict so it can be JSON-serialised directly.
print_metrics()   prints a compact summary to stdout.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred, y_prob, class_names):
    """
    Compute a full set of classification metrics.

    Parameters
    ----------
    y_true : array-like of int
    y_pred : array-like of int
    y_prob : array-like of shape (n_samples, n_classes)   softmax / predict_proba output
    class_names : list[str]

    Returns
    -------
    dict with keys:
        accuracy, f1_macro, f1_weighted,
        precision_macro, recall_macro,
        per_class  (dict keyed by class name),
        confusion_matrix  (list-of-lists)
    """
    per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_p  = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_r  = recall_score(y_true, y_pred, average=None, zero_division=0)

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "f1":        float(per_f1[i]) if i < len(per_f1) else 0.0,
            "precision": float(per_p[i])  if i < len(per_p)  else 0.0,
            "recall":    float(per_r[i])  if i < len(per_r)  else 0.0,
        }

    return {
        "accuracy":        float(accuracy_score(y_true, y_pred)),
        "f1_macro":        float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
        "f1_weighted":     float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro":    float(recall_score(y_true, y_pred, average="macro",    zero_division=0)),
        "per_class":       per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def print_metrics(metrics, class_names):
    """Print a compact metrics summary to stdout."""
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  F1  macro  : {metrics['f1_macro']:.4f}")
    print(f"  F1  weighted: {metrics['f1_weighted']:.4f}")
    print(f"  Precision  : {metrics['precision_macro']:.4f}")
    print(f"  Recall     : {metrics['recall_macro']:.4f}")
    print()
    print(f"  {'Class':<8}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}")
    print(f"  {'-'*36}")
    for name in class_names:
        m = metrics["per_class"].get(name, {})
        print(
            f"  {name:<8}  {m.get('f1', 0):.4f}  "
            f"{m.get('precision', 0):.4f}  {m.get('recall', 0):.4f}"
        )
