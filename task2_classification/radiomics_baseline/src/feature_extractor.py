"""
PyRadiomics feature extractor wrapper.

Any configuration is read once in __init__; call extract() once per patient.

The label binarisation step converts the multi-class segmentation mask
(0 = BG, 1 = Liver, 2 = Tumour) to a binary mask containing only the ROI
specified by config['data']['label_value'] before passing it to PyRadiomics.
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

        fe_cfg           = config.get("feature_extraction", {})
        self._label_value = config["data"]["label_value"]

        # Construct extractor with no args, then apply settings directly.
        # Passing a flat dict as a positional arg triggers pykwalify schema
        # validation which rejects top-level settings keys.
        self._extractor = _fe.RadiomicsFeatureExtractor()
        s = self._extractor.settings
        s["additionalInfo"] = False          # exclude diagnostic metadata keys
        s["binWidth"]       = fe_cfg.get("bin_width",       25)
        s["normalize"]      = fe_cfg.get("normalize",       False)
        s["normalizeScale"] = fe_cfg.get("normalize_scale", 100)
        # We pass a binarised mask (values 0/1), so label=1 regardless of the
        # original multi-class label value stored in self._label_value.
        s["label"] = 1

        resampling = fe_cfg.get("resampling", {})
        if resampling.get("enabled", True):
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

    def extract(self, image_path, mask_path):
        """
        Extract radiomics features for one patient.

        Parameters
        ----------
        image_path : str | Path
            Path to the CT NIfTI file (.nii or .nii.gz).
        mask_path  : str | Path
            Path to the segmentation mask NIfTI file.

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
            binary_arr = (mask_arr == self._label_value).astype(np.uint8)

            if binary_arr.sum() == 0:
                logging.warning(
                    "No voxels with label=%d found in %s — patient skipped.",
                    self._label_value, mask_path,
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
