"""
Dataset Preparation Script for Task 2: Tumor ROI Classification

Processes tumor ROI NIfTI files for training and evaluation.
Students use this script to resample, window, and convert to numpy arrays.

Pipeline per sample (unified for train and test):
  1. Read ROI NIfTI file (mask already applied, background=-1000 HU).
  2. Resample to isotropic spacing (default 1x1x1 mm) in physical space.
  3. Apply abdomen HU windowing (center=40, width=400 -> [-160, 240]).
     Note: Background (-1000 HU) is clipped to window floor (-160).
  4. Convert to numpy and save as .npy.

Expected input layout (ROI files):
    Option 1 (with roi_data/ subdirectory):
        <input_path>/
            roi_data/    <- ROI NIfTI files (*.nii.gz)
    Option 2 (flat structure, ROI files directly in input_path):
        <input_path>/
            case_1.nii.gz, case_2.nii.gz, ... <- ROI files

Usage (train with class organization):
    python prepare_dataset_for_task2.py \\
        --input_path  /data/train \\
        --output_path /data/processed_train \\
        --labels_csv  /data/labels.csv

Usage (test with flat output):
    python prepare_dataset_for_task2.py \\
        --input_path  /data/test \\
        --output_path /data/processed_test \\
        --test-mode

CSV format for train mode (required columns):
    patient_id  - base filename without extension (e.g. "patient_001")
    type        - tumor class label (e.g. "HCC", "ICC", "CRLM", "BCLM", "HH")
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
    resampler.SetDefaultPixelValue(-1000.0)  # Use background HU value for out-of-bounds voxels
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
    roi_path       = task["roi_path"]
    out_dir        = task["out_dir"]
    target_spacing = task["target_spacing"]
    window_center  = task["window_center"]
    window_width   = task["window_width"]
    output_format  = task["output_format"]
    png_axis       = task["png_axis"]

    try:
        # Process ROI NIfTI file
        roi = process_cropped_roi(
            roi_path,
            target_spacing=target_spacing,
            window_center=window_center,
            window_width=window_width,
        )
        save_sample(roi, out_dir, patient_id, output_format=output_format, png_axis=png_axis)
        shape = roi.shape
        return {"patient_id": patient_id, "split": split, "tumor_type": tumor_type,
                "status": _OK, "message": str(shape), "shape": shape}
    except Exception as exc:
        return {"patient_id": patient_id, "split": split, "tumor_type": tumor_type,
                "status": _FAIL, "message": str(exc), "shape": None}


def process_cropped_roi(
    roi_nifti_path: str,
    target_spacing: tuple = (1.0, 1.0, 1.0),
    window_center: int = 40,
    window_width: int = 400,
) -> np.ndarray:
    """Process a tumor ROI NIfTI file (used for both train and test).

    Pipeline:
      1. Read ROI NIfTI file (mask already applied, background=-1000 HU).
      2. Resample to *target_spacing* in physical space (preserves spatial metadata).
      3. Apply abdomen HU window: clip to [center-width/2, center+width/2].
         Note: Background voxels at -1000 HU are clipped to window floor.
      4. Convert to numpy array (z, y, x axis order).

    Args:
        roi_nifti_path: Path to ROI NIfTI file.
        target_spacing: Isotropic voxel spacing to resample to (mm).
        window_center:  Abdomen HU window center (default 40).
        window_width:   Abdomen HU window width (default 400 -> [-160, 240]).

    Returns:
        3-D float32 numpy array of the windowed ROI, axis order (z, y, x).
    """
    # --- Read ROI ---
    roi_sitk = sitk.ReadImage(roi_nifti_path, sitk.sitkFloat32)

    # --- Resample to target spacing ---
    roi_resampled = resample_to_spacing(roi_sitk, target_spacing, sitk.sitkLinear)

    # --- Convert to numpy (z, y, x ordering) ---
    roi_arr = sitk.GetArrayFromImage(roi_resampled).astype(np.float32)

    # --- Abdomen HU window ---
    roi_windowed = apply_abdomen_window(roi_arr, window_center, window_width)

    return roi_windowed





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
    output_format: str = "npy",
    png_axis: int = 2,
    num_workers: int = 1,
) -> None:
    """Process training ROI NIfTI files and organize by class.

    Expected input layout (either format):

        Option 1 (with roi_data/ subdirectory):
            <input_path>/
                roi_data/    <- ROI NIfTI files (*.nii.gz)
        Option 2 (flat, files directly in input_path):
            <input_path>/
                case_1.nii.gz, case_2.nii.gz, ...

    Output layout::

        <output_path>/
            <type_A>/    <- class names from CSV
                <patient_id>.npy
            <type_B>/
                ...

    Args:
        input_path:     Directory containing roi_data/ or ROI files directly.
        output_path:    Root directory for processed output (organized by class).
        labels_csv:     Path to CSV with columns ``patient_id`` and ``type``.
        target_spacing: Isotropic voxel spacing to resample to (mm).
        window_center:  Abdomen HU window center.
        window_width:   Abdomen HU window width.
        output_format:  ``'npy'`` to save a single numpy array per sample, or
                        ``'png'`` to save each slice as a grayscale PNG image.
        png_axis:       Axis to slice along when output_format is ``'png'``
                        (0=axial, 1=coronal, 2=sagittal).
        num_workers:    Number of parallel worker processes (default 1 = sequential).
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

    # --- Locate NIfTI files (check roi_data/, else use input_path directly) ---
    images_dir = os.path.join(input_path, "roi_data")
    if os.path.isdir(images_dir):
        logger.info(f"Found roi_data/ subdirectory: {images_dir}")
    elif os.path.isdir(input_path):
        logger.info(f"Using input_path directly for ROI files: {input_path}")
        images_dir = input_path
    else:
        logger.error(f"Invalid input_path: {input_path}")
        sys.exit(1)

    roi_files = [
        f for f in os.listdir(images_dir)
        if f.endswith(".nii.gz") or f.endswith(".nii")
    ]

    if not roi_files:
        logger.error(f"No NIfTI files in {images_dir}")
        sys.exit(1)

    # Create output class sub-directories up front (not safe to do inside workers)
    for cls in all_types:
        os.makedirs(os.path.join(output_path, cls), exist_ok=True)

    # --- Build task list ---
    tasks: list[dict] = []

    for roi_fname in sorted(roi_files):
        patient_id = roi_fname.replace(".nii.gz", "").replace(".nii", "")

        if patient_id not in patient_map:
            logger.warning(f"'{patient_id}' not in CSV — skipping.")
            continue

        tumor_type = patient_map[patient_id]

        tasks.append({
            "patient_id":     patient_id,
            "split":          "",
            "tumor_type":     tumor_type,
            "roi_path":       os.path.join(images_dir, roi_fname),
            "out_dir":        os.path.join(output_path, tumor_type),
            "target_spacing": target_spacing,
            "window_center":  window_center,
            "window_width":   window_width,
            "output_format":  output_format,
            "png_axis":       png_axis,
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
    output_format: str = "npy",
    png_axis: int = 2,
    num_workers: int = 1,
) -> None:
    """Process test ROI NIfTI files (flat output, no class organization).

    Expected input layout (either format):

        Option 1:  <input_path>/roi_data/ <- ROI NIfTI files
        Option 2:  <input_path>/ <- ROI files directly (case_1.nii.gz, case_2.nii.gz, ...)

    Output layout (flat structure)::

        <output_path>/
            <patient_id>.npy
            <patient_id>.npy
            ...

    Args:
        input_path:     Directory containing roi_data/ or ROI files directly.
        output_path:    Output directory where processed .npy files are saved (flat).
        target_spacing: Isotropic voxel spacing to resample to (mm).
        window_center:  Abdomen HU window center.
        window_width:   Abdomen HU window width.
        output_format:  ``'npy'`` or ``'png'`` (see prepare_dataset for details).
        png_axis:       Axis to slice along when output_format is ``'png'``.
        num_workers:    Number of parallel worker processes.
    """
    # --- Locate NIfTI files (check roi_data/, else use input_path directly) ---
    images_dir = os.path.join(input_path, "roi_data")
    if os.path.isdir(images_dir):
        logger.info(f"Found roi_data/ subdirectory: {images_dir}")
    elif os.path.isdir(input_path):
        logger.info(f"Using input_path directly for ROI files: {input_path}")
        images_dir = input_path
    else:
        logger.error(f"Invalid input_path: {input_path}")
        sys.exit(1)

    roi_files = sorted([
        f for f in os.listdir(images_dir)
        if f.endswith(".nii.gz") or f.endswith(".nii")
    ])

    if not roi_files:
        logger.error(f"No NIfTI files found in {images_dir}")
        sys.exit(1)

    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Processing test ROIs (flat output)...")
    logger.info(f"Input:  {images_dir}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Total ROIs: {len(roi_files)}\n")

    tasks = []
    for roi_fname in roi_files:
        patient_id = roi_fname.replace(".nii.gz", "").replace(".nii", "")
        roi_path = os.path.join(images_dir, roi_fname)
        tasks.append({
            "patient_id":     patient_id,
            "split":          "test",
            "tumor_type":     "roi",           # placeholder (not used for flat output)
            "roi_path":       roi_path,        # Path to ROI NIfTI file
            "out_dir":        output_path,     # Flat output directory
            "target_spacing": target_spacing,
            "window_center":  window_center,
            "window_width":   window_width,
            "output_format":  output_format,
            "png_axis":       png_axis,
        })

    total_saved = total_failed = 0
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_one, t): t for t in tasks}
            with tqdm(total=len(tasks), unit="sample", desc="Processing") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    _log_result(result, output_format, png_axis)
                    if result["status"] == _OK:
                        total_saved += 1
                    else:
                        total_failed += 1
                    pbar.update(1)
    else:
        for task in tqdm(tasks, unit="sample", desc="Processing"):
            result = _process_one(task)
            _log_result(result, output_format, png_axis)
            if result["status"] == _OK:
                total_saved += 1
            else:
                total_failed += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"Done.  Saved: {total_saved}  |  Failed/Skipped: {total_failed}")
    logger.info(f"Processed ROIs saved to: {output_path}")
    logger.info(f"Pass <output_path> to inference via: python evaluate.py --input {output_path} ...")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Task-2 classification dataset: resample -> window -> crop -> save .npy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_path", required=True,
        help="Directory containing ROI NIfTI files. "
             "Expected: (1) <input_path>/roi_data/, or "
             "(2) <input_path>/ with NIfTI files directly. "
             "Train mode (with --labels_csv): organizes outputs by class. "
             "Test mode (--test-mode): outputs flat structure.",
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
        help="Process test data (outputs saved as flat structure, no class organization). "
             "Use this for test ROI files; outputs are stored directly in <output_path>/ "
             "without class subdirectories. Requires no --labels_csv.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.test_mode:
        # ---- Test mode: test ROI files ----
        logger.info("Running in TEST MODE (processing ROI NIfTI files).")
        prepare_test_dataset(
            input_path     = args.input_path,
            output_path    = args.output_path,
            target_spacing = tuple(args.target_spacing),
            window_center  = args.window_center,
            window_width   = args.window_width,
            output_format  = args.output_format,
            png_axis       = args.png_axis,
            num_workers    = args.num_workers,
        )
    else:
        # ---- Normal mode: class-labelled training/val data ----
        if args.labels_csv is None:
            logger.error("--labels_csv is required for normal (training) mode. "
                         "Use --test-mode for test ROI files.")
            sys.exit(1)
        prepare_dataset(
            input_path     = args.input_path,
            output_path    = args.output_path,
            labels_csv     = args.labels_csv,
            target_spacing = tuple(args.target_spacing),
            window_center  = args.window_center,
            window_width   = args.window_width,
            output_format  = args.output_format,
            png_axis       = args.png_axis,
            num_workers    = args.num_workers,
        )


if __name__ == "__main__":
    main()
