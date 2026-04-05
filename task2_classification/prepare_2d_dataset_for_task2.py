"""
2D Slice Dataset Preparation for Task 2

Converts 3D NIfTI ROI volumes to 2D PNG slices for 2D model training.
Each patient is stored in a folder with selected slices.

Slice Filtering:
  All strategies automatically exclude empty slices (background only) and prefer 
  non-empty slices. Empty slices are only included if insufficient non-empty 
  slices exist to meet the target count.

Usage (train mode with labels):
    python prepare_2d_dataset_for_task2.py \\
        --input_path /path/to/roi_data \\
        --output_path /path/to/2d_dataset \\
        --labels_csv /path/to/labels.csv \\
        --slice_strategy "middle_n" \\
        --num_slices 5 \\
        --num_workers 8

Usage (test mode without labels):
    python prepare_2d_dataset_for_task2.py \\
        --input_path /path/to/test_rois \\
        --output_path /path/to/2d_test_dataset \\
        --test-mode \\
        --slice_strategy "all_nonempty" \\
        --num_workers 8

Slice Selection Strategies:
  - "all_nonempty"  : All non-empty slices (most data, recommended)
  - "middle_n"      : Middle N non-empty slices (balanced, recommended; use --num_slices)
  - "center_single" : Single center non-empty slice (fastest)
  - "equidistant"   : N equally spaced non-empty slices (balanced distribution; use --num_slices)
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def apply_abdomen_window(volume: np.ndarray, window_center: int = 40, window_width: int = 400) -> np.ndarray:
    """Apply HU windowing for abdomen soft-tissue visualization."""
    lower = window_center - window_width // 2   # -160
    upper = window_center + window_width // 2   # 240
    return np.clip(volume, lower, upper).astype(np.float32)


def normalize_to_uint8(roi: np.ndarray) -> np.ndarray:
    """Normalize windowed ROI to uint8 [0, 255] for PNG storage."""
    lower = -160.0
    upper = 240.0
    arr = np.clip(roi, lower, upper)
    arr = (arr - lower) / (upper - lower)  # [0.0, 1.0]
    return (arr * 255.0).astype(np.uint8)


def load_nifti_roi(roi_path: str) -> np.ndarray:
    """
    Load NIfTI ROI file.
    
    Returns:
        3D array with shape (D, H, W) - axial slices along first axis
    """
    import nibabel as nib
    roi_sitk = nib.load(roi_path)
    roi_arr = roi_sitk.get_fdata().astype(np.float32)
    return roi_arr


def get_nonempty_slices(volume: np.ndarray) -> list:
    """
    Get list of axial slice indices that contain non-zero content.
    
    Returns:
        List of indices along depth axis (first dimension)
    """
    nonempty = []
    for z in range(volume.shape[0]):
        if np.sum(volume[z] > 0) > 0:
            nonempty.append(z)
    return nonempty


def select_slices(volume: np.ndarray, strategy: str = "middle_n", num_slices: int = 5) -> list:
    """
    Select axial slices from 3D volume based on strategy, preferring non-empty slices.
    
    Args:
        volume: 3D array with shape (D, H, W) where:
                D = superior-inferior (Z-axis, axial slices)
                H = anterior-posterior (Y-axis)
                W = left-right (X-axis)
        strategy: "all_nonempty", "middle_n", "center_single", "equidistant"
        num_slices: Number of slices for "middle_n" and "equidistant"
        
    Returns:
        List of axial slice indices to extract (non-empty when possible)
    """
    depth = volume.shape[0]
    all_slices = list(range(depth))
    nonempty_slices = get_nonempty_slices(volume)
    
    if strategy == "all_nonempty":
        # All non-empty slices, fallback to all if none found
        return nonempty_slices if nonempty_slices else all_slices
    
    elif strategy == "middle_n":
        # Middle N non-empty slices centered around volume center
        if not nonempty_slices:
            # Fallback if all empty
            center = depth // 2
            half_range = num_slices // 2
            start = max(0, center - half_range)
            end = min(depth, center + half_range + 1)
            return list(range(start, end))
        
        # Find center of non-empty region
        center_idx = len(nonempty_slices) // 2
        half_range = num_slices // 2
        start_idx = max(0, center_idx - half_range)
        end_idx = min(len(nonempty_slices), center_idx + half_range + 1)
        selected = nonempty_slices[start_idx:end_idx]
        
        # If we don't have enough non-empty slices, fill with empty ones
        if len(selected) < num_slices:
            empty_slices = [s for s in all_slices if s not in nonempty_slices]
            selected.extend(empty_slices[:num_slices - len(selected)])
        
        return sorted(selected)[:num_slices]
    
    elif strategy == "center_single":
        # Single center non-empty slice, fallback to center slice
        if nonempty_slices:
            return [nonempty_slices[len(nonempty_slices) // 2]]
        else:
            return [depth // 2]
    
    elif strategy == "equidistant":
        # N equally spaced non-empty slices, fill with empty if needed
        if not nonempty_slices:
            # Fallback if all empty
            if num_slices >= depth:
                return all_slices
            indices = np.linspace(0, depth - 1, num_slices, dtype=int).tolist()
            return sorted(set(indices))
        
        if num_slices >= len(nonempty_slices):
            # Use all non-empty, then fill with empty slices
            selected = nonempty_slices[:]
            if len(selected) < num_slices:
                empty_slices = [s for s in all_slices if s not in nonempty_slices]
                selected.extend(empty_slices[:num_slices - len(selected)])
            return sorted(selected)[:num_slices]
        
        # Select equidistant from non-empty slices
        indices = np.linspace(0, len(nonempty_slices) - 1, num_slices, dtype=int)
        return sorted([nonempty_slices[i] for i in indices])
    
    else:
        logger.warning(f"Unknown strategy '{strategy}', defaulting to 'middle_n'")
        return select_slices(volume, strategy="middle_n", num_slices=num_slices)


def process_single_roi(task: dict) -> dict:
    """Process a single ROI file."""
    patient_id = task["patient_id"]
    roi_path = task["roi_path"]
    out_dir = task["out_dir"]
    window_center = task["window_center"]
    window_width = task["window_width"]
    slice_strategy = task["slice_strategy"]
    num_slices = task["num_slices"]
    
    try:
        # Load and preprocess
        roi = load_nifti_roi(roi_path)
        roi = apply_abdomen_window(roi, window_center, window_width)
        
        # Select slices
        slice_indices = select_slices(roi, strategy=slice_strategy, num_slices=num_slices)
        
        # Extract and save slices as PNG
        patient_out_dir = os.path.join(out_dir, patient_id)
        os.makedirs(patient_out_dir, exist_ok=True)
        
        for idx, z in enumerate(slice_indices):
            slice_2d = roi[z]  # Extract axial slice (Z-axis): shape (H, W)
            slice_uint8 = normalize_to_uint8(slice_2d)
            img = Image.fromarray(slice_uint8, mode="L")
            fname = f"{str(idx).zfill(3)}.png"
            img.save(os.path.join(patient_out_dir, fname))
        
        return {
            "patient_id": patient_id,
            "status": "OK",
            "num_slices": len(slice_indices),
            "message": f"Saved {len(slice_indices)} slices",
        }
    
    except Exception as e:
        return {
            "patient_id": patient_id,
            "status": "FAIL",
            "num_slices": 0,
            "message": str(e),
        }


def prepare_2d_dataset(
    input_path: str,
    output_path: str,
    labels_csv: str = None,
    window_center: int = 40,
    window_width: int = 400,
    slice_strategy: str = "middle_n",
    num_slices: int = 5,
    num_workers: int = 1,
    test_mode: bool = False,
) -> None:
    """
    Process ROI data and extract 2D slices.
    
    Args:
        input_path: Directory containing NIfTI ROI files
        output_path: Root directory for output
        labels_csv: CSV with patient_id and type columns (required if not test_mode)
        window_center: HU window center
        window_width: HU window width
        slice_strategy: Strategy for slice selection
        num_slices: Number of slices for certain strategies
        num_workers: Number of parallel workers
        test_mode: If True, process without labels CSV
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Locate ROI files
    images_dir = os.path.join(input_path, "roi_data")
    if os.path.isdir(images_dir):
        logger.info(f"Found roi_data/ subdirectory: {images_dir}")
    elif os.path.isdir(input_path):
        logger.info(f"Using input_path directly: {input_path}")
        images_dir = input_path
    else:
        logger.error(f"Invalid input_path: {input_path}")
        sys.exit(1)
    
    # Find ROI files
    roi_files = [
        f for f in os.listdir(images_dir)
        if f.endswith((".nii.gz", ".nii"))
    ]
    
    if not roi_files:
        logger.error(f"No ROI files found in {images_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(roi_files)} ROI files")
    
    # Load labels if not test mode
    patient_map = {}
    if not test_mode:
        if not labels_csv or not os.path.isfile(labels_csv):
            logger.error("labels_csv is required for non-test mode")
            sys.exit(1)
        
        df = pd.read_csv(labels_csv)
        if not all(col in df.columns for col in ["patient_id", "type"]):
            logger.error("CSV must contain 'patient_id' and 'type' columns")
            sys.exit(1)
        
        patient_map = dict(zip(df["patient_id"].astype(str), df["type"].astype(str)))
        logger.info(f"Loaded labels for {len(patient_map)} patients")
    
    # Build task list
    tasks = []
    for roi_fname in sorted(roi_files):
        patient_id = roi_fname.replace(".nii.gz", "").replace(".nii", "")
        
        if not test_mode and patient_id not in patient_map:
            logger.warning(f"'{patient_id}' not in CSV — skipping")
            continue
        
        tasks.append({
            "patient_id": patient_id,
            "roi_path": os.path.join(images_dir, roi_fname),
            "out_dir": output_path,
            "window_center": window_center,
            "window_width": window_width,
            "slice_strategy": slice_strategy,
            "num_slices": num_slices,
        })
    
    if not tasks:
        logger.error("No valid samples to process")
        sys.exit(1)
    
    logger.info(f"Processing {len(tasks)} ROIs with strategy '{slice_strategy}'")
    if slice_strategy in ["middle_n", "equidistant"]:
        logger.info(f"  Extracting {num_slices} slices per ROI")
    
    # Process tasks
    total_ok = 0
    total_fail = 0
    total_slices = 0
    
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_roi, t): t for t in tasks}
            with tqdm(total=len(tasks), unit="ROI", desc="Extracting slices") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result["status"] == "OK":
                        total_ok += 1
                        total_slices += result["num_slices"]
                    else:
                        total_fail += 1
                        logger.warning(f"  {result['patient_id']}: {result['message']}")
                    pbar.update(1)
    else:
        with tqdm(total=len(tasks), unit="ROI", desc="Extracting slices") as pbar:
            for task in tasks:
                result = process_single_roi(task)
                if result["status"] == "OK":
                    total_ok += 1
                    total_slices += result["num_slices"]
                else:
                    total_fail += 1
                    logger.warning(f"  {result['patient_id']}: {result['message']}")
                pbar.update(1)
    
    logger.info(f"\nCompleted:")
    logger.info(f"  Success: {total_ok} ROIs")
    logger.info(f"  Failed:  {total_fail} ROIs")
    logger.info(f"  Total slices: {total_slices}")
    logger.info(f"  Output directory: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract 2D slices from 3D NIfTI ROIs for 2D model training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_path", required=True,
        help="Directory containing NIfTI ROI files or roi_data/ subdirectory",
    )
    parser.add_argument(
        "--output_path", required=True,
        help="Output directory for 2D dataset (organized by patient_id/)",
    )
    parser.add_argument(
        "--labels_csv", default=None,
        help="CSV with patient_id and type columns (required for training mode)",
    )
    parser.add_argument(
        "--test-mode", action="store_true",
        help="Process test data (no labels CSV required)",
    )
    parser.add_argument(
        "--window_center", type=int, default=40,
        help="HU window center for abdomen soft-tissue",
    )
    parser.add_argument(
        "--window_width", type=int, default=400,
        help="HU window width (clips to [center-width/2, center+width/2])",
    )
    parser.add_argument(
        "--slice_strategy", 
        choices=["all_nonempty", "middle_n", "center_single", "equidistant"],
        default="middle_n",
        help="Strategy for selecting slices from each volume",
    )
    parser.add_argument(
        "--num_slices", type=int, default=5,
        help="Number of slices for 'middle_n' and 'equidistant' strategies",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help="Number of parallel worker processes",
    )
    
    args = parser.parse_args()
    
    prepare_2d_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        labels_csv=args.labels_csv,
        window_center=args.window_center,
        window_width=args.window_width,
        slice_strategy=args.slice_strategy,
        num_slices=args.num_slices,
        num_workers=args.num_workers,
        test_mode=args.test_mode,
    )


if __name__ == "__main__":
    main()
