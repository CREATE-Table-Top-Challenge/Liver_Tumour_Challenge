![CREATE Challenge 2026 Banner](assets/banner.png)

# 🏥 CREATE Table-Top Challenge 2026

## 📚 Overview

Welcome to the **CREATE Table-Top Challenge 2026**! This challenge focuses on developing automated methods for:

- **Sub-task 1 (Segmentation):** Accurately segment liver and tumours on 3D abdominal CT images
- **Sub-task 2 (Classification):** Predict tumour type (HCC, ICC, CRLM, BCLM, HH) from CT ROIs

Both tasks use **PyTorch + MONAI**, with baseline models and training templates provided.

---

## 🚀 Quick Start

### For School of Computing GPU Server
✅ **Everything is pre-configured for you!**
- Python environment ready with all dependencies
- Dataset pre-downloaded and prepared
- Skip to → [Training the Networks](#-training-the-networks)

### For Local Development
Follow the setup steps below to configure your environment and download data.

---

## 🔧 Environment Setup

### Option 1: Local Development (Windows/Linux/Mac)

#### Step 1: Create Virtual Environment
```bash
# Navigate to the repository
cd src

# Create virtual environment
python -m venv main
source main/bin/activate              # Linux/Mac
# OR
main\Scripts\activate                 # Windows PowerShell
```

#### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** The `requirements.txt` includes all dependencies for both Task 1 and Task 2 (PyTorch, MONAI, nibabel, etc.)

#### Step 3: Verify Installation
```bash
python -c 'import torch; import monai; print("Setup complete!")'
```

### Option 2: Radiomics Baseline (Optional, Separate Environment)

If you want to try the radiomics feature extraction approach for Task 2:

```bash
cd task2_classification/radiomics_baseline

# Create separate Python 3.9 environment (required for PyRadiomics)
python3.9 -m venv radiomics_env
source radiomics_env/bin/activate
pip install --upgrade pip
pip install "numpy<2"
pip install pyradiomics scikit-learn xgboost pandas joblib PyYAML tqdm

# Follow instructions in radiomics_baseline/README.md
```

**Note:** Radiomics uses a separate environment due to Python 3.9 requirement. See [radiomics_baseline/README.md](task2_classification/radiomics_baseline/README.md) for details.

---

## 📥 Dataset Preparation

### Download Data

**Training Data** → Available during challenge start
- CT images: `imagesTr/` directory
- Segmentation masks: `labelsTr/` directory
- Labels CSV: `train.csv` (maps patient IDs to types)

**Test Data** → Available on test data release date
- CT images only (unlabeled)

### Data Structure
```
raw_data/
├── imagesTr/            # 3D CT scans (.nii.gz)
│   ├── case_001.nii.gz
│   ├── case_002.nii.gz
│   └── ...
├── labelsTr/            # Segmentation masks (.nii.gz)
│   ├── case_001.nii.gz
│   ├── case_002.nii.gz
│   └── ...
└── train.csv       # Patient → Tumour Type mapping
```

**Labels in segmentation masks:**
- `0` = Background
- `1` = Liver
- `2` = Tumour

**Tumour types in CSV:**
- HCC (Hepatocellular carcinoma)
- ICC (Intrahepatic cholangiocarcinoma)
- CRLM (Colorectal liver metastases)
- BCLM (Breast cancer liver metastases)
- HH (Hemangioma)

---

## 🔄 Task 1: Abdominal CT Image Segmentation

### Objective
Build a 3D segmentation model that labels each voxel as Background (0), Liver (1), or Tumour (2).

### Evaluation Metric
- **Primary:** Dice Similarity Coefficient (DSC) — higher is better
- **Secondary:** Hausdorff Distance 95th percentile (HD95) — lower is better

### Quick Start

#### Train a Model
```bash
cd task1_segmentation

# Using configuration file (recommended)
python train.py --config baseline_config.yaml

# OR specify paths directly (with separate validation set):
python train.py \
    --train_images /path/to/train/imagesTr \
    --train_labels /path/to/train/labelsTr \
    --val_images /path/to/val/imagesTr \
    --val_labels /path/to/val/labelsTr \
    --max_epochs 100 \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --output_dir ./checkpoints
```

#### Key Arguments

| Argument | Description |
|---|---|
| `--config` | Configuration file with all settings |
| `--train_images` | Path to training CT images |
| `--train_labels` | Path to training segmentation masks |
| `--val_images` | Path to validation CT images (or use `--val_fraction`) |
| `--val_labels` | Path to validation segmentation masks |
| `--val_fraction` | Automatically split fraction from training as val (if `--val_images` not provided) |
| `--max_epochs` | Number of training epochs |
| `--batch_size` | Batch size per GPU |
| `--learning_rate` | Initial learning rate |
| `--weight_decay` | Weight decay for regularization |
| `--num_workers` | Number of data loading workers |
| `--val_interval` | Validation frequency (epochs) |
| `--patience` | Early stopping patience |
| `--output_dir` | Directory to save checkpoints and logs |
| `--num_classes` | Number of classes (Background, Liver, Tumour) |
| `--seed` | Random seed for reproducibility |
| `--compute_hd95` | Compute HD95 metric during validation |
| `--resume_from` | Path to checkpoint to resume training from |

#### Evaluate on Test Data
```bash
python evaluate.py \
    --checkpoint ./checkpoints/baseline_unet/best_model.pth \
    --test_images /path/to/test/imagesTr \
    --output_dir ./predictions \
    --group 1
```

#### Key Arguments for Evaluation

| Argument | Description |
|---|---|
| `--checkpoint` | Path to saved model checkpoint |
| `--test_images` | Path to test CT images |
| `--test_labels` | Path to test segmentation masks (optional, for metrics only) |
| `--batch_size` | Batch size for inference |
| `--num_workers` | Number of data loading workers |
| `--output_dir` | Directory to save predictions |
| `--num_classes` | Override number of classes from checkpoint |
| `--group` | Group number for output ZIP filename (e.g., 1 → `group1_task1_results.zip`) |

### Output

- **Predictions:** NIfTI files with predicted segmentation masks
- **ZIP File Submission:** `group<N>_task1_results.zip` containing predicted masks for test cases.
- **Logs:** Training curves and detailed metrics

> ⚠️ **Important:** Task 1 predictions are used to extract ROIs for Task 2. Save your predicted masks - you'll need them for Task 2 test data preprocessing (see [Task 2: Prepare Test Data](#step-1-prepare-test-data-roi-extraction)).

---

## 🎯 Task 2: Liver Tumour Type Prediction

### Objective
Classify tumour type (HCC, ICC, CRLM, BCLM, HH) from cropped tumour ROIs extracted using segmentation masks.

### Evaluation Metric
- **Primary:** Macro-averaged F1 score — higher is better
- **Secondary:** Macro-averaged AUROC — higher is better

### Quick Start

#### Step 1: Prepare Training Data (ROI Extraction)
```bash
cd task2_classification

# Extract tumour ROIs from training data using ground truth masks
python prepare_dataset_for_task2.py \
    --input_path /path/to/raw_data \
    --output_path /path/to/processed_data \
    --labels_csv /path/to/train.csv \
    --target_label 2 \
    --bbox_margin 5 \
    --num_workers 8
```

**This creates:**
```
processed_data/
├── HCC/          # Tumour ROIs (.npy files)
├── ICC/
├── CRLM/
├── BCLM/
└── HH/
```

#### Step 1b: Prepare Test Data (ROI Extraction using Task 1 Predictions)

**Important workflow:** Task 1 predicted segmentation masks are used to extract ROIs for test data:

```bash
cd task2_classification

# Extract tumour ROIs from test data using PREDICTED masks from Task 1
# Note: Use --test-mode flag for test data (no labels CSV)
python prepare_dataset_for_task2.py \
    --input_path /path/to/test_data \
    --output_path /path/to/test_processed_data \
    --target_label 2 \
    --bbox_margin 5 \
    --num_workers 8 \
    --test-mode
```

> 📌 **Note:** The test ROI extraction uses predicted segmentation masks from Task 1 (store them in `labelsTr/` directory alongside test images). The `--test-mode` flag indicates test data processing (no labels CSV required). Ensure your Task 1 model predictions are saved in NIfTI format (in `labelsTr/` directory) in the same directory as the test CT images.

#### Key Arguments for Preprocessing

| Argument | Description |
|---|---|
| `--input_path` | Path containing `imagesTr/` and `labelsTr/` subdirectories |
| `--output_path` | Root directory where processed ROIs will be saved |
| `--labels_csv` | CSV file with columns `patient_id` and `type` (required for training; not used with `--test-mode`) |
| `--test-mode` | Flag to process unlabelled test data (no CSV required) |
| `--target_label` | Mask label to isolate before computing bounding box (e.g., 2 for tumour) |
| `--target_spacing` | Resample to isotropic spacing (mm) |
| `--window_center` | HU window center for CT clipping (should match `transforms.py` values) |
| `--window_width` | HU window width (should match `transforms.py` values; range = [center - width/2, center + width/2]) |
| `--bbox_margin` | Extra voxel margin around bounding box on each side |
| `--num_workers` | Number of parallel worker processes |
| `--output_format` | Output format: `npy` or `png` |
| `--png_axis` | Axis to slice along for PNG output (0=axial, 1=coronal, 2=sagittal) |

#### Step 2: Train Classifier
```bash
# Using configuration file (recommended)
python train.py --config baseline_config.yaml
```

#### Key Arguments for Training

| Argument | Description |
|---|---|
| `--config` | Configuration file |
| `--data_dir` | Path to processed ROIs (with class subdirs: HCC/, ICC/, etc.) |
| `--val_dir` | Separate validation directory (if not provided, uses `--train_val_split`) |
| `--model_type` | Model architecture: `resnet18`, `resnet50`, or `densenet121` |
| `--num_classes` | Number of output classes |
| `--pretrained` | Use ImageNet pretrained weights (set to `true`) |
| `--max_epochs` | Max epochs per fold |
| `--batch_size` | Batch size per GPU |
| `--learning_rate` | Initial learning rate |
| `--weight_decay` | Weight decay for regularization |
| `--num_workers` | Number of data loading workers |
| `--val_interval` | Validation frequency (epochs) |
| `--patience` | Early stopping patience |
| `--k_folds` | Number of cross-validation folds |
| `--train_val_split` | Train/val split ratio if no separate `--val_dir` |
| `--output_dir` | Directory to save checkpoints and logs |
| `--experiment_name` | Experiment name for logging |
| `--seed` | Random seed for reproducibility |
| `--resume_from` | Path to checkpoint to resume training from |

#### Step 3: Evaluate on Test Set
```bash
python evaluate.py \
    --config baseline_config.yaml \
    --input ./test_processed_data \
    --output ./predictions
```

#### Key Arguments for Evaluation

| Argument | Description |
|---|---|
| `--config` | Configuration file with model settings |
| `--input` | Path to test ROIs directory (output of `prepare_dataset_for_task2.py --test-mode`) |
| `--output` | Directory where `predictions.csv` will be saved |
| `--models-dir` | Directory containing `cv_results.json` (ensemble) or `best_model.pth` (single model) |
| `--model` | Explicit path to single `.pth` checkpoint (overrides `--models-dir`) |
| `--model-type` | Model architecture (from config or override) |
| `--num-classes` | Number of output classes (from config or override) |
| `--classes` | Class names in order (from config or override) |
| `--device` | Device to use: `cuda` or `cpu` |
| `--group` | Group number for output filename (e.g., 1 → `group1_task2_results.csv`) |

### Output

- **Predictions:** `group<N>_task2_results.csv` with predicted labels and probabilities
- **Logs:** Training curves and fold-specific metrics
- **Cross-validation results:** `cv_results.json` (if using k-fold)

---

## 📊 Project Structure

```
src/
├── README.md                          # This file
├── requirements.txt                   # Unified dependencies for both tasks
├── task1_segmentation/
│   ├── train.py                       # Training script
│   ├── evaluate.py                    # Evaluation script
│   ├── baseline_config.yaml           # Baseline configuration
│   ├── src/
│   │   ├── model.py                   # Model factory
│   │   ├── unet_model.py              # MONAI 3D UNet
│   │   ├── segresnet_model.py         # MONAI SegResNet
│   │   ├── swinunetr_model.py         # MONAI SwinUNETR
│   │   ├── trainer.py                 # Training loop
│   │   ├── data_loader.py             # Data loading
│   │   ├── transforms.py              # Data augmentation
│   │   └── metric_tracker.py          # Metrics logging
│   └── checkpoints/                   # Saved models (generated)
|
├── task2_classification/
│   ├── train.py                       # Training with k-fold CV
│   ├── evaluate.py                    # Evaluation script
│   ├── prepare_dataset_for_task2.py   # ROI extraction
│   ├── baseline_config.yaml           # Baseline configuration
│   ├── src/
│   │   ├── model.py                   # Model factory
│   │   ├── resnet18_model.py          # MONAI 3D ResNet-18
│   │   ├── resnet50_model.py          # MONAI 3D ResNet-50
│   │   ├── densenet121_model.py       # MONAI 3D DenseNet-121
│   │   ├── trainer.py                 # Training with CV support
│   │   ├── data_loader.py             # ROI dataset loader
│   │   ├── transforms.py              # Augmentation transforms
│   │   ├── inferer.py                 # Inference utilities
│   │   └── metrics.py                 # Classification metrics
│   ├── radiomics_baseline/            # Alternative: Radiomics + Random Forest
│   │   ├── README.md                  # Setup for radiomics environment
│   │   └── ...
|   └── checkpoints/                   # Saved models (generated)
```

---

## 📤 Submission

### Workflow Overview

**Tasks 1 and 2 are independent during training.** Task 1 predictions are only needed for preprocessing Task 2 test data:

```
Training Phase (Parallel):
├─ Train Task 1 model (see Task 1: Quick Start)
└─ Train Task 2 model (see Task 2: Quick Start)

Evaluation & Submission Phase (Sequential):
├─ Evaluate Task 1 on test data → generates predictions
├─ Use Task 1 predictions to preprocess Task 2 test data
├─ Evaluate Task 2 on test ROIs
└─ Submit both results
```

### Submission Files

**Task 1 Submission:**
- **File:** `group<N>_task1_results.zip`
- **Contents:** Predicted segmentation masks (NIfTI format) for all test cases

**Task 2 Submission:**
- **File:** `group<N>_task2_results.csv`
- **Contents:** Predictions with class labels and probabilities for all test cases

**See earlier sections for detailed commands:**
- [Task 1: Quick Start](#-task-1-abdominal-ct-image-segmentation) for training and evaluation
- [Task 2: Quick Start](#-task-2-liver-tumour-type-prediction) for training and evaluation

---

## 📚 Additional Resources

- **PyTorch Docs:** https://pytorch.org/docs/
- **MONAI Docs:** https://docs.monai.io/
---

## ✅ Checklist Before Submission

- [ ] Task 1: Model trained and checkpoint saved
- [ ] Task 1: Predictions generated for test set
- [ ] Task 1: `group<N>_task1_results.zip` created
- [ ] Task 2: ROIs extracted from test data
- [ ] Task 2: Model trained and checkpoint saved
- [ ] Task 2: Predictions generated for test set
- [ ] Task 2: `group<N>_task2_results.csv` created
- [ ] Both tasks: Predictions in correct format and directory
---

**Good luck! 🚀**  
