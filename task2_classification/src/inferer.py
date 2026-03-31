"""
Task 2: Liver Tumour Classification -- Inference Library

Core inference functions used by evaluate.py.
Import this module rather than running it directly.

Functions
---------
load_model              -- load a single .pth checkpoint
load_ensemble_models    -- load all fold models via cv_results.json manifest
load_roi_scan           -- read .npy or NIfTI file as numpy array
preprocess_roi          -- apply val transforms, return batch tensor
predict_roi             -- single-model softmax inference
predict_ensemble        -- multi-model probability averaging
save_csv                -- write predictions.csv
setup_logging           -- configure logging with optional file handler
"""
import csv
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Make src/ importable when this module is imported from the project root
_this_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_this_dir))

from base_model import build_model
from transforms import get_val_transforms


def setup_logging(output_dir=None):
    handlers = [logging.StreamHandler()]
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(output_dir) / 'inference.log'))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=handlers,
    )


def load_model(checkpoint_path, model_type='resnet18', num_classes=3, device='cpu'):
    """Load a single checkpoint and return an eval-mode model.

    Supports both legacy checkpoints (state dict saved from a bare MONAI network,
    keys without a 'net.' prefix) and new-style checkpoints saved from a
    ClassificationModelBase wrapper (keys prefixed with 'net.').
    """
    logging.info(f"Loading {model_type} model from {checkpoint_path}")

    model = build_model(
        config={'model': {'model_type': model_type}},
        num_classes=num_classes,
        model_type=model_type,
    )

    ckpt  = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt

    # Legacy checkpoints were saved from a bare MONAI model (no wrapper), so
    # their keys have no 'net.' prefix.  Load into model.net in that case.
    first_key = next(iter(state))
    if first_key.startswith('net.'):
        model.load_state_dict(state)
    else:
        model.net.load_state_dict(state)

    model.to(device).eval()
    logging.info("Model loaded successfully")
    return model


def load_roi_scan(scan_path):
    """Load .npy or NIfTI scan as float32 numpy array."""
    if scan_path.endswith('.npy'):
        return np.load(scan_path).astype(np.float32)
    if scan_path.endswith('.nii.gz') or scan_path.endswith('.nii'):
        import nibabel as nib
        return nib.load(scan_path).get_fdata().astype(np.float32)
    raise ValueError(f"Unsupported file format: {scan_path}")


def preprocess_roi(roi_data, transforms, device):
    """Apply val transforms and return a batch tensor (1, C, D, H, W)."""
    t = transforms(roi_data) if transforms is not None else torch.from_numpy(roi_data).float()
    if t.ndim == 3:
        t = t.unsqueeze(0)
    return t.unsqueeze(0).to(device)


def predict_roi(model, roi_tensor):
    """Single-model inference. Returns (predicted_class_idx, probabilities)."""
    with torch.no_grad():
        probs = torch.softmax(model(roi_tensor), dim=1).squeeze(0).cpu().numpy()
    return int(np.argmax(probs)), probs


def load_ensemble_models(cv_manifest_path: str, model_type: str, num_classes: int, device: str):
    """
    Load all fold models listed in a cv_results.json manifest.
    Model paths in the manifest are stored relative to the manifest file
    Returns a list of models in eval mode, one per fold.
    """
    import json
    manifest_dir = os.path.dirname(os.path.abspath(cv_manifest_path))
    with open(cv_manifest_path, encoding='utf-8') as f:
        manifest = json.load(f)

    models = []
    for entry in manifest['folds']:
        fold = entry['fold']
        rel_path = entry['model_path']
        # Resolve relative path from the manifest's directory
        path = os.path.join(manifest_dir, rel_path) if not os.path.isabs(rel_path) else rel_path
        if not os.path.isfile(path):
            logging.warning(f"Fold {fold} model not found at {path}, skipping.")
            continue
        logging.info(f"Loading fold {fold} model from {path}")
        m = load_model(path, model_type=model_type, num_classes=num_classes, device=device)
        models.append(m)

    if not models:
        raise RuntimeError("No fold models could be loaded from CV manifest.")
    logging.info(f"Loaded {len(models)} fold model(s) for ensemble inference.")
    return models


def predict_ensemble(models: list, roi_tensor: torch.Tensor):
    """
    Run inference with an ensemble of models.

    Averages the softmax probabilities across all fold models
    (probability averaging — equivalent to a soft vote weighted by confidence).
    This is the standard best-practice for competition ensembles.

    Returns:
        Tuple of (predicted_class, averaged_probabilities)
    """
    all_probs = []
    with torch.no_grad():
        for model in models:
            output = model(roi_tensor)
            probs = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()
            all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)   # shape: (num_classes,)
    predicted_class = int(np.argmax(avg_probs))
    return predicted_class, avg_probs


def save_csv(rows, class_names, output_path):
    """
    Save predictions as CSV.

    Columns: patient_id, predicted_class, prob_<class0>, prob_<class1>, ...
    """
    prob_cols  = [f'prob_{c.replace(" ", "_")}' for c in class_names]
    fieldnames = ['patient_id', 'predicted_class'] + prob_cols
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logging.info(f"[OK] Predictions saved to: {output_path}")


