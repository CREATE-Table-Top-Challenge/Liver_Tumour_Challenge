"""
Dataset Preparation Script for Task 2: Tumor ROI Classification

Processes NIfTI CT volumes and segmentation masks into pre-cropped numpy arrays
organized by tumor type (class) for training/evaluation.

Pipeline per sample:
  1. Resample CT and mask to isotropic spacing (default 1x1x1 mm)
  2. Apply abdomen HU windowing (center=40, width=400 -> [-160, 240])
  3. Crop CT to mask bounding box (ROI)
  4. Save cropped array as .npy in output/<type>/

Expected input layout:
    <input_path>/
        imagesTr/    <- CT volumes  (*.nii.gz)
        labelsTr/    <- Mask files  (*.nii.gz)

Usage:
    python prepare_dataset_for_task2.py \\
        --input_path  /path/to/raw_dataset \\
        --output_path /path/to/processed_dataset \\
        --labels_csv    /path/to/split_info.csv

CSV format (required columns):
    patient_id  - base filename without extension (e.g. "patient_001")
    type        - tumor class label (e.g. "liver", "cyst", "tumor")
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import SimpleITK as sitk
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def resample_to_spacing(
    image: sitk.Image,
    target_spacing: tuple,
    interpolator=sitk.sitkLinear,
) -> sitk.Image:
    """Resample a SimpleITK image to *target_spacing* in physical space.

    SimpleITK handles the full spatial transform (origin, direction, spacing)
    so resampling is geometrically correct even for oblique acquisitions.

    Args:
        image:          Input SimpleITK image.
        target_spacing: Desired spacing in mm for each axis (x, y, z).
        interpolator:   SimpleITK interpolator constant.
                        Use ``sitk.sitkLinear`` for CT,
                        ``sitk.sitkNearestNeighbor`` for label masks.

    Returns:
        Resampled SimpleITK image.
    """
    original_spacing = image.GetSpacing()   # (x, y, z)
    original_size    = image.GetSize()      # (x, y, z)

    new_size = [
        int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(interpolator)

    return resampler.Execute(image)


def apply_abdomen_window(volume: np.ndarray, window_center: int = 40, window_width: int = 400) -> np.ndarray:
    """Clip CT Hounsfield units to the abdomen soft-tissue window.

    Args:
        volume:         CT array in HU.
        window_center:  HU center (default 40).
        window_width:   HU width  (default 400 -> range [-160, 240]).

    Returns:
        Clipped float32 array with values in [center - width/2, center + width/2].
    """
    lower = window_center - window_width // 2   # -160
    upper = window_center + window_width // 2   #  240
    return np.clip(volume, lower, upper).astype(np.float32)


def get_bounding_box_3d(
    mask_volume: np.ndarray,
) -> tuple[int, int, int, int, int, int] | None:
    """Return the 3-D tight bounding box of all nonzero voxels in *mask_volume*.

    Array axis order is (z, y, x) — consistent with SimpleITK ``GetArrayFromImage``.

    Returns:
        ``(min_x, min_y, min_z, max_x, max_y, max_z)`` or ``None`` if the
        mask is entirely zero.
    """
    if not np.any(mask_volume):
        return None

    non_zero = np.argwhere(mask_volume)          # shape (N, 3) — columns: z, y, x
    min_z, min_y, min_x = non_zero.min(axis=0)
    max_z, max_y, max_x = non_zero.max(axis=0)

    return (int(min_x), int(min_y), int(min_z), int(max_x), int(max_y), int(max_z))


def crop_to_bbox(
    volume: np.ndarray,
    mask: np.ndarray,
    margin: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop *volume* and *mask* to the tight 3-D bounding box of the ROI.

    Array axis order must be (z, y, x) for both inputs.

    Args:
        volume: 3-D CT array (z, y, x).
        mask:   3-D binary/label mask — any nonzero value is treated as ROI.
        margin: Extra voxels added on every side (clamped to array bounds).

    Returns:
        ``(cropped_volume, cropped_mask)`` — both clipped to the same box.
        If the mask is empty, returns the originals unchanged.
    """
    bbox = get_bounding_box_3d(mask)
    if bbox is None:
        logger.warning("Empty mask — returning full volume without cropping.")
        return volume, mask

    min_x, min_y, min_z, max_x, max_y, max_z = bbox

    # Apply margin, clamped to array bounds  (axis order: z, y, x)
    min_z = max(0, min_z - margin)
    max_z = min(volume.shape[0] - 1, max_z + margin)
    min_y = max(0, min_y - margin)
    max_y = min(volume.shape[1] - 1, max_y + margin)
    min_x = max(0, min_x - margin)
    max_x = min(volume.shape[2] - 1, max_x + margin)

    cropped_volume = volume[min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1]
    cropped_mask   = mask  [min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1]

    return cropped_volume, cropped_mask

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

# HU window bounds used during preprocessing (must stay in sync with apply_abdomen_window defaults)
_HU_MIN = -160.0   # window_center(40) - window_width(400) / 2
_HU_MAX =  240.0   # window_center(40) + window_width(400) / 2


def roi_to_uint8(roi: np.ndarray) -> np.ndarray:
    """Normalise a pre-windowed ROI array to uint8 [0, 255].

    Assumes HU values are already clipped to [_HU_MIN, _HU_MAX] by
    apply_abdomen_window().  Maps that range linearly to [0, 255].
    """
    arr = np.clip(roi, _HU_MIN, _HU_MAX)
    arr = (arr - _HU_MIN) / (_HU_MAX - _HU_MIN)   # [0.0, 1.0]
    return (arr * 255.0).astype(np.uint8)


def save_as_npy(roi: np.ndarray, out_dir: str, patient_id: str) -> None:
    """Save the 3-D ROI as a single .npy file."""
    out_path = os.path.join(out_dir, patient_id + ".npy")
    np.save(out_path, roi)


def save_as_png_slices(
    roi: np.ndarray,
    out_dir: str,
    patient_id: str,
    axis: int = 0,
) -> None:
    """Save each slice of the 3-D ROI as a separate grayscale PNG.

    The array returned by the SimpleITK pipeline has axis order (z, y, x):
      axis=0 -> axial   (z slices)  [default]
      axis=1 -> coronal (y slices)
      axis=2 -> sagittal(x slices)

    Each file is named ``<patient_id>_slice_<NNN>.png``.

    Args:
        roi:        3-D float32 array with HU values in [_HU_MIN, _HU_MAX].
        out_dir:    Directory in which PNG files are written.
        patient_id: Base identifier embedded in every filename.
        axis:       Axis to slice along (0=axial, 1=coronal, 2=sagittal).
    """
    uint8_roi = roi_to_uint8(roi)
    n_slices = uint8_roi.shape[axis]
    pad = len(str(n_slices - 1))   # zero-pad width

    for idx in range(n_slices):
        slice_2d = np.take(uint8_roi, idx, axis=axis)   # 2-D (H, W) or (H, D) etc.
        img = Image.fromarray(slice_2d, mode="L")
        fname = f"{patient_id}_slice_{str(idx).zfill(pad)}.png"
        img.save(os.path.join(out_dir, fname))


def save_sample(
    roi: np.ndarray,
    out_dir: str,
    patient_id: str,
    output_format: str = "npy",
    png_axis: int = 2,
) -> None:
    """Dispatch to the correct saver based on *output_format*.

    Args:
        roi:           3-D float32 ROI array.
        out_dir:       Output class directory (already created).
        patient_id:    Patient identifier used in filenames.
        output_format: ``'npy'`` (default) or ``'png'``.
        png_axis:      Axis to slice along when output_format is ``'png'``.
    """
    if output_format == "png":
        save_as_png_slices(roi, out_dir, patient_id, axis=png_axis)
    else:
        save_as_npy(roi, out_dir, patient_id)


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def find_file(directory: str, patient_id: str) -> str | None:
    """Locate the NIfTI file for *patient_id* inside *directory*.

    Tries common suffixes: .nii.gz and .nii.
    The file stem must equal *patient_id* (case-insensitive).
    """
    for ext in (".nii.gz", ".nii"):
        candidate = os.path.join(directory, patient_id + ext)
        if os.path.isfile(candidate):
            return candidate
    # Fallback: scan directory for any file whose basename starts with patient_id
    for fname in os.listdir(directory):
        stem = fname.replace(".nii.gz", "").replace(".nii", "")
        if stem == patient_id:
            return os.path.join(directory, fname)
    return None


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

# Result status constants used by the parallel worker
_OK   = "ok"
_SKIP = "skip"
_FAIL = "fail"


def _process_one(task: dict) -> dict:
    """Top-level worker executed in a subprocess.

    Accepts a single *task* dict so it is easily picklable.  Returns a result
    dict with keys: ``patient_id``, ``split``, ``tumor_type``, ``status``
    (one of _OK / _SKIP / _FAIL), ``message``, and ``shape``.
    """
    patient_id     = task["patient_id"]
    split          = task["split"]
    tumor_type     = task["tumor_type"]
    ct_path        = task["ct_path"]
    mask_path      = task["mask_path"]
    out_dir        = task["out_dir"]
    target_spacing = task["target_spacing"]
    window_center  = task["window_center"]
    window_width   = task["window_width"]
    bbox_margin    = task["bbox_margin"]
    output_format  = task["output_format"]
    png_axis       = task["png_axis"]
    target_label   = task["target_label"]

    try:
        roi = process_sample(
            ct_path, mask_path,
            target_spacing=target_spacing,
            window_center=window_center,
            window_width=window_width,
            bbox_margin=bbox_margin,
            target_label=target_label,
        )
        save_sample(roi, out_dir, patient_id, output_format=output_format, png_axis=png_axis)
        shape = roi.shape
        return {"patient_id": patient_id, "split": split, "tumor_type": tumor_type,
                "status": _OK, "message": str(shape), "shape": shape}
    except Exception as exc:
        return {"patient_id": patient_id, "split": split, "tumor_type": tumor_type,
                "status": _FAIL, "message": str(exc), "shape": None}


def process_sample(
    ct_path: str,
    mask_path: str,
    target_spacing: tuple = (1.0, 1.0, 1.0),
    window_center: int = 40,
    window_width: int = 400,
    bbox_margin: int = 0,
    target_label: int | None = None,
) -> np.ndarray:
    """Load one CT / mask pair with SimpleITK, preprocess, and return the cropped ROI.

    Pipeline:
      1. Read CT and mask with SimpleITK (preserves full spatial metadata).
      2. Cast CT to float32; cast mask to uint8 (nearest-neighbour interp).
      3. Resample both to *target_spacing* in physical space using the same
         geometric reference — CT with linear interpolation, mask with
         nearest-neighbour so label values are preserved exactly.
      4. Convert to numpy arrays (axis order: z, y, x).
      5. Apply abdomen HU window to CT.
      6. Build ROI mask: if *target_label* is given, isolate that label;
         otherwise binarise all nonzero voxels.
      7. Crop CT to the ROI bounding box.

    Args:
        ct_path:        Path to the CT NIfTI file.
        mask_path:      Path to the segmentation mask NIfTI file.
        target_spacing: Isotropic voxel spacing to resample to (mm).
        window_center:  Abdomen HU window center.
        window_width:   Abdomen HU window width.
        bbox_margin:    Extra voxel margin around the bounding box on each side.
        target_label:   Integer label value to isolate from the mask before
                        computing the bounding box (e.g. 4 for tumour in a
                        5-class liver mask).  ``None`` (default) uses all
                        nonzero voxels.

    Returns:
        3-D float32 numpy array of the cropped ROI, axis order (z, y, x).
    """
    # --- Read ---
    ct_sitk   = sitk.ReadImage(ct_path,   sitk.sitkFloat32)
    mask_sitk = sitk.ReadImage(mask_path, sitk.sitkUInt8)

    # --- Resample to isotropic spacing in physical space ---
    ct_resampled   = resample_to_spacing(ct_sitk,   target_spacing, sitk.sitkLinear)
    mask_resampled = resample_to_spacing(mask_sitk, target_spacing, sitk.sitkNearestNeighbor)

    # --- Convert to numpy (z, y, x ordering) ---
    ct_arr   = sitk.GetArrayFromImage(ct_resampled).astype(np.float32)   # (z, y, x)
    mask_arr = sitk.GetArrayFromImage(mask_resampled)                     # (z, y, x)

    # --- Build ROI mask ---
    if target_label is not None:
        mask_binary = (mask_arr == target_label).astype(np.uint8)
        if mask_binary.sum() == 0:
            raise ValueError(
                f"Label {target_label} not found in mask after resampling. "
                f"Present labels: {np.unique(mask_arr).tolist()}"
            )
    else:
        mask_binary = (mask_arr > 0).astype(np.uint8)
        if mask_binary.sum() == 0:
            raise ValueError("Mask contains no nonzero voxels after resampling.")

    # --- Abdomen HU window ---
    ct_windowed = apply_abdomen_window(ct_arr, window_center, window_width)

    # --- Crop CT and mask to the ROI bounding box ---
    roi, roi_mask = crop_to_bbox(ct_windowed, mask_binary, margin=bbox_margin)

    # --- Zero out voxels outside the mask within the cropped box ---
    # -1000 HU ~= air, used as a universal background value for CT scans
    background_hu = -1000.0
    roi = roi.copy()
    roi[roi_mask == 0] = background_hu

    return roi


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def prepare_dataset(
    input_path: str,
    output_path: str,
    labels_csv: str,
    target_spacing: tuple = (1.0, 1.0, 1.0),
    window_center: int = 40,
    window_width: int = 400,
    bbox_margin: int = 0,
    output_format: str = "npy",
    png_axis: int = 2,
    num_workers: int = 1,
    target_label: int | None = None,
) -> None:
    """Process a dataset and save numpy arrays organised by class.

    Expected input layout::

        <input_path>/
            imagesTr/    <- CT volumes  (*.nii.gz)
            labelsTr/    <- Mask files  (*.nii.gz)

    Output layout::

        <output_path>/
            <type_A>/    <- class names from CSV
                <patient_id>.npy
            <type_B>/
                ...

    Args:
        input_path:     Directory containing ``imagesTr/`` and ``labelsTr/``.
        output_path:    Root directory for processed output.
        labels_csv:       Path to CSV with columns ``patient_id`` and ``type``.
        target_spacing:  Isotropic voxel spacing to resample to (mm).
        window_center:   Abdomen HU window center.
        window_width:    Abdomen HU window width.
        bbox_margin:     Extra voxels added around bounding box on each side.
        output_format:   ``'npy'`` to save a single numpy array per sample, or
                         ``'png'`` to save each slice as a grayscale PNG image.
        png_axis:        Axis to slice along when output_format is ``'png'``
                         (0=axial, 1=coronal, 2=sagittal).
        num_workers:     Number of parallel worker processes (default 1 = sequential).
        target_label:    Integer mask label to isolate before computing the
                         bounding box (e.g. ``4`` for tumour in a 5-class
                         liver segmentation mask).  ``None`` (default) uses
                         all nonzero voxels.
    """
    # --- Load CSV ---
    if not os.path.isfile(labels_csv):
        logger.error(f"CSV file not found: {labels_csv}")
        sys.exit(1)

    df = pd.read_csv(labels_csv)
    required_cols = {"patient_id", "type"}
    if not required_cols.issubset(df.columns):
        logger.error(f"CSV must contain columns: {required_cols}. Found: {list(df.columns)}")
        sys.exit(1)

    # Map patient_id -> type (class label)
    patient_map: dict[str, str] = dict(zip(df["patient_id"].astype(str), df["type"].astype(str)))
    all_types = sorted(df["type"].unique())
    logger.info(f"Classes found in CSV: {all_types}")
    logger.info(f"Total patients in CSV: {len(patient_map)}")

    # --- Locate imagesTr / labelsTr directly inside input_path ---
    images_dir = os.path.join(input_path, "imagesTr")
    labels_dir = os.path.join(input_path, "labelsTr")

    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        logger.error(
            f"Expected imagesTr/ and labelsTr/ directly inside: {input_path}\n"
            f"Found subdirs: {os.listdir(input_path)}"
        )
        sys.exit(1)

    image_files = [
        f for f in os.listdir(images_dir)
        if f.endswith(".nii.gz") or f.endswith(".nii")
    ]

    if not image_files:
        logger.error(f"No NIfTI files in {images_dir}")
        sys.exit(1)

    # Create output class sub-directories up front (not safe to do inside workers)
    for cls in all_types:
        os.makedirs(os.path.join(output_path, cls), exist_ok=True)

    # --- Build task list ---
    tasks: list[dict] = []

    for img_fname in sorted(image_files):
        patient_id = img_fname.replace(".nii.gz", "").replace(".nii", "")

        if patient_id not in patient_map:
            logger.warning(f"'{patient_id}' not in CSV — skipping.")
            continue

        tumor_type = patient_map[patient_id]
        mask_path  = find_file(labels_dir, patient_id)

        if mask_path is None:
            logger.warning(f"No mask found for '{patient_id}' — skipping.")
            continue

        tasks.append({
            "patient_id":     patient_id,
            "split":          "",
            "tumor_type":     tumor_type,
            "ct_path":        os.path.join(images_dir, img_fname),
            "mask_path":      mask_path,
            "out_dir":        os.path.join(output_path, tumor_type),
            "target_spacing": target_spacing,
            "window_center":  window_center,
            "window_width":   window_width,
            "bbox_margin":    bbox_margin,
            "output_format":  output_format,
            "png_axis":       png_axis,
            "target_label":   target_label,
        })

    if not tasks:
        logger.error("No valid samples found. Check paths and CSV.")
        sys.exit(1)

    logger.info(f"\nTotal samples to process: {len(tasks)}  |  workers: {num_workers}")

    # --- Dispatch tasks (parallel or sequential) ---
    total_saved  = 0
    total_failed = 0

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_one, t): t for t in tasks}
            with tqdm(total=len(tasks), unit="sample", desc="Processing", leave=False) as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    _log_result(result, output_format, png_axis)
                    if result["status"] == _OK:
                        total_saved += 1
                    else:
                        total_failed += 1
                    pbar.set_postfix(saved=total_saved, failed=total_failed)
                    pbar.update(1)
    else:
        for task in tqdm(tasks, unit="sample", desc="Processing", leave=False):
            result = _process_one(task)
            _log_result(result, output_format, png_axis)
            if result["status"] == _OK:
                total_saved += 1
            else:
                total_failed += 1

    # --- Summary ---
    logger.info(f"\n{'='*60}")
    logger.info(f"Done.  Saved: {total_saved}  |  Failed/Skipped: {total_failed}")
    logger.info(f"Output directory: {output_path}")


def _log_result(result: dict, output_format: str, png_axis: int) -> None:
    """Log the outcome of a single worker result."""
    pid  = result["patient_id"]
    splt = result["split"]
    typ  = result["tumor_type"]
    if result["status"] == _OK:
        shape = result["shape"]
        fmt_info = f"{shape[png_axis]} PNG slices" if output_format == "png" else f"shape {shape}"
        logger.info(f"  [{splt}/{typ}] {pid} — {fmt_info} -> saved")
    else:
        logger.error(f"  [{splt}/{typ}] {pid} — FAILED: {result['message']}")


def prepare_test_dataset(
    input_path: str,
    output_path: str,
    target_spacing: tuple = (1.0, 1.0, 1.0),
    window_center: int = 40,
    window_width: int = 400,
    bbox_margin: int = 0,
    output_format: str = "npy",
    png_axis: int = 2,
    num_workers: int = 1,
    target_label: int | None = None,
) -> None:
    """
    Process test data that has no class labels.

    Same preprocessing pipeline as ``prepare_dataset`` (resample, window, crop)
    but does not require a CSV file.  All outputs are saved directly to
    the output directory as a flat list of samples::

        <output_path>/<patient_id>.npy

    The resulting flat folder is then passed directly to ``src/inferer.py``
    via ``--input``.

    Expected input layout::

        <input_path>/
            imagesTr/    <- CT volumes  (*.nii.gz)
            labelsTr/    <- ROI masks   (*.nii.gz)  (needed for bounding box crop)

    Args:
        input_path:     Directory containing imagesTr/ and labelsTr/.
        output_path:    Root output directory.  Processed arrays will be saved here
        Other args identical to :func:`prepare_dataset`.
    """
    images_dir = os.path.join(input_path, "imagesTr")
    labels_dir = os.path.join(input_path, "labelsTr")

    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        logger.error(
            f"Expected imagesTr/ and labelsTr/ subdirectories inside: {input_path}"
        )
        sys.exit(1)

    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.endswith(".nii.gz") or f.endswith(".nii")
    ])

    if not image_files:
        logger.error(f"No NIfTI files found in {images_dir}")
        sys.exit(1)

    out_dir = output_path
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Test output directory: {out_dir}")
    logger.info(f"Processing {len(image_files)} test case(s)...")

    tasks = []
    for img_fname in image_files:
        patient_id = img_fname.replace(".nii.gz", "").replace(".nii", "")
        mask_path  = find_file(labels_dir, patient_id)
        if mask_path is None:
            logger.warning(f"No mask found for '{patient_id}' — skipping.")
            continue
        tasks.append({
            "patient_id":     patient_id,
            "split":          "test",
            "tumor_type":     "unknown",      # placeholder — not used in output
            "ct_path":        os.path.join(images_dir, img_fname),
            "mask_path":      mask_path,
            "out_dir":        out_dir,         # flat output, no class subfolder
            "target_spacing": target_spacing,
            "window_center":  window_center,
            "window_width":   window_width,
            "bbox_margin":    bbox_margin,
            "output_format":  output_format,
            "png_axis":       png_axis,
            "target_label":   target_label,
        })

    total_saved = total_failed = 0
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_one, t): t for t in tasks}
            with tqdm(total=len(tasks), unit="sample", desc="Test preprocessing") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    _log_result(result, output_format, png_axis)
                    if result["status"] == _OK:
                        total_saved += 1
                    else:
                        total_failed += 1
                    pbar.update(1)
    else:
        for task in tqdm(tasks, unit="sample", desc="Test preprocessing"):
            result = _process_one(task)
            _log_result(result, output_format, png_axis)
            if result["status"] == _OK:
                total_saved += 1
            else:
                total_failed += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"Done.  Saved: {total_saved}  |  Failed/Skipped: {total_failed}")
    logger.info(f"Test ROIs saved to: {output_path}")
    logger.info(f"Pass this directory to the inference script via --input {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Task-2 classification dataset: resample -> window -> crop -> save .npy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_path", required=True,
        help="Directory containing imagesTr/ and labelsTr/ subdirectories. "
             "Normal mode: must contain imagesTr/ and labelsTr/ directly (no split sub-folders). "
             "Test mode (--test-mode): same layout, no CSV required.",
    )
    parser.add_argument(
        "--output_path", required=True,
        help="Root directory where processed numpy arrays will be saved.",
    )
    parser.add_argument(
        "--labels_csv", default=None,
        help="CSV file with columns 'patient_id' and 'type' (tumor class). "
             "Required for normal mode; not used with --test-mode.",
    )
    parser.add_argument(
        "--test-mode", action="store_true",
        help="Process unlabelled test data (no CSV required). "
             "--input_path should point to a folder containing imagesTr/ and labelsTr/. "
             "Outputs saved to <output_path>/test/ as a flat list of .npy files, "
             "ready for src/inferer.py --input.",
    )
    parser.add_argument(
        "--target_spacing", nargs=3, type=float, default=[1.0, 1.0, 1.0],
        metavar=("X", "Y", "Z"),
        help="Target isotropic voxel spacing in mm.",
    )
    parser.add_argument(
        "--window_center", type=int, default=40,
        help="Abdomen HU window center.",
    )
    parser.add_argument(
        "--window_width", type=int, default=400,
        help="Abdomen HU window width (lower = center - width/2, upper = center + width/2).",
    )
    parser.add_argument(
        "--bbox_margin", type=int, default=0,
        help="Extra voxel margin added around the mask bounding box on each side.",
    )
    parser.add_argument(
        "--output_format", choices=["npy", "png"], default="npy",
        help="Output format: 'npy' saves one .npy file per sample; "
             "'png' saves each slice as a grayscale PNG named <patient_id>_slice_NNN.png.",
    )
    parser.add_argument(
        "--png_axis", type=int, default=0, choices=[0, 1, 2],
        help="Axis to slice along when --output_format png is used. "
             "Array axes after SimpleITK resampling are (z, y, x): "
             "0=axial (default), 1=coronal, 2=sagittal.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help="Number of parallel worker processes. 1 = sequential (default). "
             "Set to os.cpu_count() or a fixed value for faster processing.",
    )
    parser.add_argument(
        "--target_label", type=int, default=None,
        help="Mask label integer to isolate before computing the bounding box "
             "(e.g. 4 for tumour in a 5-class liver mask). "
             "Omit or set to None to use all nonzero voxels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.test_mode:
        # ---- Test mode: no labels, flat output ----
        logger.info("Running in TEST MODE (no class labels required).")
        prepare_test_dataset(
            input_path     = args.input_path,
            output_path    = args.output_path,
            target_spacing = tuple(args.target_spacing),
            window_center  = args.window_center,
            window_width   = args.window_width,
            bbox_margin    = args.bbox_margin,
            output_format  = args.output_format,
            png_axis       = args.png_axis,
            num_workers    = args.num_workers,
            target_label   = args.target_label,
        )
    else:
        # ---- Normal mode: class-labelled training/val data ----
        if args.labels_csv is None:
            logger.error("--labels_csv is required for normal (training) mode. "
                         "Use --test-mode for unlabelled test data.")
            sys.exit(1)
        prepare_dataset(
            input_path     = args.input_path,
            output_path    = args.output_path,
            labels_csv       = args.labels_csv,
            target_spacing = tuple(args.target_spacing),
            window_center  = args.window_center,
            window_width   = args.window_width,
            bbox_margin    = args.bbox_margin,
            output_format  = args.output_format,
            png_axis       = args.png_axis,
            num_workers    = args.num_workers,
            target_label   = args.target_label,
        )


if __name__ == "__main__":
    main()
