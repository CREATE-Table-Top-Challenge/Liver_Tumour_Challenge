# PyRadiomics Baseline — Task 2 Liver Tumour Classification

Radiomics-based classifier using extracted image texture features and classical ML (Random Forest, SVM, XGBoost, etc.).

**⚠️ Python 3.9 Required:**  
PyRadiomics has significant compatibility issues with Python 3.10+. Use **Python 3.9.x** for this baseline.

---

## Setup

### 1. Create Python 3.9 Virtual Environment

**Windows (PowerShell):**
```powershell
python3.9 -m venv radiomics_env
.\radiomics_env\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python3.9 -m venv radiomics_env
source radiomics_env/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install "numpy<2"
pip install pyradiomics scikit-learn xgboost pandas joblib PyYAML tqdm
```

This installs:
- **PyRadiomics 3.0.1+** — texture feature extraction
- **scikit-learn** — classifiers (Random Forest, SVM, Logistic Regression, MLP)
- **XGBoost** (optional) — for gradient boosting classifier
- Core dependencies: pandas, numpy, SimpleITK, PyYAML, tqdm, joblib

---

## Workflow

### Step 1: Extract Features (Train Data)

Extract PyRadiomics texture features from CT images + segmentation masks:

```bash
python extract_features.py \
    --config config.yaml \
    --images /path/to/dataset_train/imagesTr \
    --masks  /path/to/dataset_train/labelsTr \
    --labels-csv /path/to/labels.csv \
    --output-csv features/train_features.csv
```

**Arguments:**
- `--images` — Directory with CT images (`*.nii.gz` or `*.nii`)
- `--masks` — Directory with segmentation masks (same patient IDs as images) (`*.nii.gz` or `*.nii`)
- `--labels-csv` — CSV mapping patient_id → class (required for train data)
- `--output-csv` — Output CSV file with extracted features + class labels

**Expected output:**  
`features/train_features.csv` with columns: `patient_id`, feature columns (1000+), `class`

### Step 2: Train Classifier

Train a scikit-learn classifier on the extracted features:

```bash
python train.py \
    --config config.yaml \
    --features features/train_features.csv
```

**CLI Options:**
```bash
--classifier {random_forest|svm|logistic_regression|xgboost|mlp}
--k-folds N           # 0 or 1 = single split, N≥2 = k-fold CV
--output-dir DIR      # Override checkpoint directory
--seed SEED           # Random seed (default: 42)
```

**Default:** Random Forest with 5-fold cross-validation

**Expected output:**  
```
model_checkpoints/radiomics_rf_baseline/
├── final/
│   └── pipeline.joblib          # Trained on all data (after CV)
├── fold_1/
│   ├── pipeline.joblib
│   ├── results.json
│   └── feature_importance.csv
├── fold_2/ ... fold_5/          # Similar for each fold
├── cv_results.json              # Overall CV metrics
└── [other fold results]
```

### Step 3: Extract Test Features

Extract features from test data (**without** class labels):

```bash
python extract_features.py \
    --config config.yaml \
    --images /path/to/dataset_test/imagesTr \
    --masks  /path/to/dataset_test/labelsTr \
    --output-csv features/test_features.csv
```

**Note:** Omit `--labels-csv` for test data. Output CSV will not include class column.

### Step 4: Evaluate & Generate Submission

Run inference on test data and generate the submission CSV:

```bash
python evaluate.py \
    --config config.yaml \
    --features features/test_features.csv \
    --models-dir model_checkpoints/radiomics_rf_baseline
```

**Alternatives:**

Single model submission:
```bash
python evaluate.py \
    --config config.yaml \
    --features features/test_features.csv \
    --model model_checkpoints/radiomics_rf_baseline/pipeline.joblib
```

**Expected output:**  
```
results/group<N>_task2_results.csv
```

Columns: `patient_id`, `predicted_class`, `prob_BCLM`, `prob_CRLM`, `prob_HCC`, `prob_HH`, `prob_ICC`

---

## Configuration

Edit `config.yaml` to customize:

```yaml
feature_extraction:
  resampling:
    target_spacing: [1.0, 1.0, 1.0]  # Isotropic voxel size (mm)
  feature_classes:
    - firstorder      # Intensity statistics
    - glcm            # Texture co-occurrence
    - glszm           # Size-zone matrix
    - glrlm           # Run-length matrix
    - ngtdm           # Grey-tone difference matrix

classifier:
  type: random_forest             # Options: random_forest, svm, xgboost, mlp
  params:
    n_estimators: 100
    max_depth: null
    random_state: 42
```

---

## Troubleshooting

**Issue: `ImportError: No module named 'radiomics'`**  
→ Activate the radiomics environment: `source radiomics_env/bin/activate`

**Issue: `ModuleNotFoundError: No module named 'xgboost'`**  
→ XGBoost is optional. Either:
  - Install: `pip install xgboost`
  - Or use a different classifier: `--classifier random_forest`

**Issue: Feature extraction hangs or crashes**  
→ Set `n_jobs: 1` in `config.yaml` on Windows (multiprocessing issues with PyRadiomics)

**Issue: Python 3.10+ errors with PyRadiomics**  
→ Use Python 3.9: Create env with `python3.9 -m venv` instead

---

## References

- **PyRadiomics Documentation:** https://pyradiomics.readthedocs.io/
- **Radiomics Paper:** https://doi.org/10.1148/radiol.2015151169
- **scikit-learn:** https://scikit-learn.org/

