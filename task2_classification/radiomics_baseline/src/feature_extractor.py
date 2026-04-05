"""
PyRadiomics feature extractor wrapper.

Provides two methods:
- extract(): For CT + segmentation mask pairs (legacy/flexible approach)
- extract_from_roi(): For tumor ROI NIfTI files (simplified approach)
"""
import logging

import numpy as np
import SimpleITK as sitk

# Suppress PyRadiomics INFO/DEBUG output — it logs per-voxel operations which
# is extremely verbose and adds measurable I/O overhead during extraction.
logging.getLogger("radiomics").setLevel(logging.WARNING)
# Suppress the "GLCM is symmetrical, therefore Sum Average = 2 * Joint Average"
# warning — it fires once per patient and adds no actionable information.
logging.getLogger("radiomics.glcm").setLevel(logging.ERROR)

# Map YAML interpolator strings to SimpleITK constants
_INTERPOLATORS = {
    "sitkLinear":              sitk.sitkLinear,
    "sitkBSpline":             sitk.sitkBSpline,
    "sitkNearestNeighbor":     sitk.sitkNearestNeighbor,
    "sitkLanczosWindowedSinc": sitk.sitkLanczosWindowedSinc,
}


class RadiomicsExtractor:
    """
    Thin wrapper around radiomics.featureextractor.RadiomicsFeatureExtractor.

    Parameters
    ----------
    config : dict
        Full config dict loaded from config.yaml.
    """

    def __init__(self, config):
        # Import here so the module can be imported even without pyradiomics
        # installed (unit tests, doc generation, etc.)
        from radiomics import featureextractor as _fe

        fe_cfg = config.get("feature_extraction", {})

        # Construct extractor with no args, then apply settings directly.
        # Passing a flat dict as a positional arg triggers pykwalify schema
        # validation which rejects top-level settings keys.
        self._extractor = _fe.RadiomicsFeatureExtractor()
        s = self._extractor.settings
        s["additionalInfo"] = False          # exclude diagnostic metadata keys
        s["binWidth"]       = fe_cfg.get("bin_width",       25)
        s["normalize"]      = fe_cfg.get("normalize",       False)
        s["normalizeScale"] = fe_cfg.get("normalize_scale", 100)
        # We pass a binarised mask (values 0/1), so label=1.
        s["label"] = 1

        resampling = fe_cfg.get("resampling", {})
        if resampling.get("enabled", False):
            # ROI data is already resampled to 1mm³, so this is usually disabled.
            s["resampledPixelSpacing"] = resampling.get(
                "target_spacing", [1.0, 1.0, 1.0]
            )
            interp_str = resampling.get("interpolator", "sitkBSpline")
            s["interpolator"] = _INTERPOLATORS.get(interp_str, sitk.sitkBSpline)

        # Enable only the requested feature families
        self._extractor.disableAllFeatures()
        for fc in fe_cfg.get("feature_classes", ["firstorder", "glcm"]):
            self._extractor.enableFeatureClassByName(fc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, image_path, mask_path, label_value=2):
        """
        Extract radiomics features for one patient from CT + mask pair.

        NOTE: This method is for backward compatibility / flexible use.
        For pre-extracted ROI files, use extract_from_roi() instead.

        Parameters
        ----------
        image_path : str | Path
            Path to the CT NIfTI file (.nii or .nii.gz).
        mask_path  : str | Path
            Path to the segmentation mask NIfTI file (multi-class).
        label_value : int
            Integer label in mask that corresponds to tumor ROI (default: 2).

        Returns
        -------
        dict[str, float]
            Feature name → scalar value.  Only numeric features are included;
            NaN / Inf values are replaced with 0.0.
        None
            Returned when the ROI is empty or extraction fails.
        """
        try:
            image = sitk.ReadImage(str(image_path))
            mask  = sitk.ReadImage(str(mask_path))

            # Binarise the multi-class mask: keep only the tumour label
            mask_arr   = sitk.GetArrayFromImage(mask)
            binary_arr = (mask_arr == label_value).astype(np.uint8)

            if binary_arr.sum() == 0:
                logging.warning(
                    "No voxels with label=%d found in %s — patient skipped.",
                    label_value, mask_path,
                )
                return None

            binary_mask = sitk.GetImageFromArray(binary_arr)
            binary_mask.CopyInformation(mask)   # preserve spacing / origin / direction

            result = self._extractor.execute(image, binary_mask)

            features = {}
            for k, v in result.items():
                if k.startswith("diagnostics_"):
                    continue
                try:
                    val = float(v)
                    features[k] = 0.0 if (np.isnan(val) or np.isinf(val)) else val
                except (TypeError, ValueError):
                    pass   # skip non-numeric entries (e.g. version strings)

            return features or None

        except Exception as exc:
            logging.error("Extraction failed for %s: %s", image_path, exc)
            return None

    def extract_from_roi(self, roi_path):
        """
        Extract radiomics features from a pre-extracted tumor ROI NIfTI file.

        The ROI is already masked, cropped, and padded to uniform size.
        A binary mask is created on-the-fly from the ROI:
          - Void/background voxels: -1024 HU (padding from organizer)
          - Tumor voxels: > -1024 HU (any value in windowed CT range)
          - Binary mask: `(roi > -1024).astype(uint8)` → 1 = tumor, 0 = background

        Parameters
        ----------
        roi_path : str | Path
            Path to pre-extracted ROI NIfTI file (.nii or .nii.gz).
            Expected: ROI in HU units with:
              - Background/void voxels: -1024 HU (padding from organizer)
              - Tumor voxels: windowed to [-160, 240] HU range

        Returns
        -------
        dict[str, float]
            Feature name → scalar value.  Only numeric features are included;
            NaN / Inf values are replaced with 0.0.
        None
            Returned when no foreground voxels detected or extraction fails.
        """
        try:
            roi = sitk.ReadImage(str(roi_path))

            # Create binary mask on-the-fly from ROI
            # Exclude void voxels (background = -1024 HU) and keep tumor voxels.
            # Binary mask: 1 for tumor, 0 for background/void
            roi_arr = sitk.GetArrayFromImage(roi)
            binary_arr = (roi_arr > -1024).astype(np.uint8)

            if binary_arr.sum() == 0:
                logging.warning("No foreground voxels found in %s — patient skipped.", roi_path)
                return None

            # Create SimpleITK binary mask and preserve spatial metadata from ROI
            binary_mask = sitk.GetImageFromArray(binary_arr)
            binary_mask.CopyInformation(roi)

            # Extract radiomics features using ROI as image and binary mask as segmentation
            result = self._extractor.execute(roi, binary_mask)

            # Parse features: exclude diagnostics, handle NaN/Inf, keep numeric only
            features = {}
            for k, v in result.items():
                if k.startswith("diagnostics_"):
                    continue
                try:
                    val = float(v)
                    features[k] = 0.0 if (np.isnan(val) or np.isinf(val)) else val
                except (TypeError, ValueError):
                    pass   # skip non-numeric entries

            return features or None

        except Exception as exc:
            logging.error("Extraction failed for %s: %s", roi_path, exc)
            return None
