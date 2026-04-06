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
- Activate the environment and skip to → [Training the Networks](#-task-1-abdominal-ct-image-segmentation)

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

**Dataset Link:** [CREATE Challenge Dataset on SharePoint](https://queensuca-my.sharepoint.com/:f:/g/personal/23ws9_queensu_ca/IgBCzxn9AJFZTo8SETBTIqhKAdf1vNFKUlaL59eH_IdI3x8?e=yonBuc)

> 📧 **Password:** Sent to your group email on challenge start date. Check your inbox!

**Dataset Structure:**
- `task1_data/` — Task 1 training data (CT images + segmentation masks)
- `task2_data/` — Task 2 training data (Tumor ROI files + labels CSV)

**Test Data** → Available on test data release date

**What's Inside:**

**Task 1 Training Data:**
- `imagesTr/` — 3D CT scans (.nii.gz)
- `labelsTr/` — Segmentation masks (.nii.gz)

**Task 2 Training Data:**
- `roi_data/` — Pre-extracted tumor ROI NIfTI files (.nii.gz, resampled to 1mm³ and padded to uniform size)
- `labels.csv` — Maps patient IDs to tumor types

### Data Structure

#### Task 1
```
task1_data/
├── imagesTr/            # 3D CT scans (.nii.gz)
│   ├── case_001.nii.gz
│   ├── case_002.nii.gz
│   └── ...
└── labelsTr/            # Segmentation masks (.nii.gz)
    ├── case_001.nii.gz
    ├── case_002.nii.gz
    └── ...
```

**Labels in segmentation masks:**
- `0` = Background
- `1` = Liver
- `2` = Tumour

#### Task 2 Training Data:
```
task2_data/
├── roi_data/            # Pre-extracted Tumor ROI NIfTI files (.nii.gz)
│   ├── case_001.nii.gz
│   ├── case_002.nii.gz
│   └── ...
└── labels.csv          # Patient ID → Tumor Type mapping
```

**Tumor types in CSV:**
- HCC (Hepatocellular carcinoma)
- ICC (Intrahepatic cholangiocarcinoma)
- CRLM (Colorectal liver metastases)
- BCLM (Breast cancer liver metastases)
- HH (Hemangioma)

---

### ⚠️ Important Data Rules

- **Task 2 ROI test data cannot be used for training Task 1.** Task 1 training must use only the full CT volumes in `task1_data/`. Any use of Task 2 test ROI data for Task 1 training will result in disqualification.

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
    --output_dir ./model_checkpoints
```

**Note on Configuration Precedence:** CLI arguments always override YAML config values. For example, `python train.py --config baseline_config.yaml --learning_rate 5e-4` will use 5e-4 from CLI, ignoring the config file's value.

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
| `--class_names` | Class names (foreground only) |
| `--seed` | Random seed for reproducibility |
| `--compute_hd95` | Compute HD95 metric during validation (default: true) |
| `--resume_from` | Path to checkpoint to resume training from |

**Configuration Philosophy:** Simple, top-level parameters are exposed as CLI args for quick experimentation. Advanced parameters (optimizer type, scheduler configuration, architecture-specific tuning) are configured in YAML only to avoid CLI clutter.

#### Evaluate on Test Data
```bash
python evaluate.py \
    --checkpoint ./model_checkpoints/baseline_unet/best_model.pth \
    --test_images /path/to/test/imagesTr \
    --output_dir ./predictions \
    --group 1
```

#### Key Arguments for Evaluation

| Argument | Description |
|---|---|
| `--checkpoint` | Path to saved model checkpoint |
| `--config` | Path to config YAML file (optional) |
| `--test_images` | Path to test CT images |
| `--test_labels` | Path to test segmentation masks (optional, for metrics only) |
| `--num_classes` | Number of segmentation classes |
| `--class_names` | Class names (foreground only) |
| `--batch_size` | Batch size for inference |
| `--num_workers` | Number of data loading workers |
| `--output_dir` | Directory to save predictions |
| `--group` | Group number for output ZIP filename (e.g., 1 → `group1_task1_results.zip`) |

### Output

- **Predictions:** NIfTI files with predicted segmentation masks
- **ZIP File Submission:** `group<N>_task1_results.zip` containing predicted masks for test cases.
- **Logs:** Training curves and detailed metrics

---

## 🎯 Task 2: Liver Tumour Type Prediction

### Objective
Classify tumour type (HCC, ICC, CRLM, BCLM, HH) from tumour ROIs.

**Note:** Pre-extracted ROI data (resampled to 1mm³ and padded to uniform size) is loaded directly from NIfTI files.

### Evaluation Metric
- **Primary:** Macro-averaged F1 score — higher is better
- **Secondary:** Macro-averaged AUROC — higher is better

### Quick Start

#### Step 1: Train Classifier
```bash
cd task2_classification

# Using configuration file (recommended)
python train.py --config baseline_config.yaml

# OR specify paths directly:
python train.py \
    --data_dir /path/to/roi_data \
    --labels_csv /path/to/labels.csv \
    --max_epochs 100 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --model_type resnet18 \
    --k_folds 5 \
    --output_dir ./model_checkpoints
```

**Note on Configuration Precedence:** CLI arguments always override YAML config values. For example, `python train.py --config baseline_config.yaml --batch_size 32` will use 32 from CLI, ignoring the config file's value.

#### Key Arguments for Training

| Argument | Description |
|---|---|
| `--config` | Configuration file |
| `--data_dir` | Path to directory containing NIfTI ROI files |
| `--labels_csv` | Path to CSV file with columns: patient_id, type |
| `--val_dir` | Separate validation data directory (if not provided, uses `--val_fraction`) |
| `--val_labels_csv` | CSV file for validation data (required if `--val_dir` is specified) |
| `--model_type` | Model architecture: `resnet18`, `resnet50`, or `densenet121` |
| `--num_classes` | Number of output classes |
| `--class_names` | Class names in order (e.g., `BCLM CRLM HCC HH ICC`) |
| `--max_epochs` | Max epochs per fold |
| `--batch_size` | Batch size per GPU |
| `--learning_rate` | Initial learning rate |
| `--weight_decay` | Weight decay for regularization |
| `--num_workers` | Number of data loading workers |
| `--val_interval` | Validation frequency (epochs) |
| `--patience` | Early stopping patience |
| `--k_folds` | Number of cross-validation folds |
| `--val_fraction` | Fraction of training data reserved for validation (e.g., 0.2 = 20% val) |
| `--output_dir` | Directory to save checkpoints and logs |
| `--experiment_name` | Experiment name for logging |
| `--seed` | Random seed for reproducibility |
| `--resume_from` | Path to checkpoint to resume training from |

**Configuration Philosophy:** Simple, top-level parameters are exposed as CLI args for quick experimentation. Advanced parameters (optimizer type, loss function, scheduler configuration, architecture-specific tuning) are configured in YAML only to avoid CLI clutter.

#### Step 2: Evaluate on Test Set
```bash
python evaluate.py \
    --config baseline_config.yaml \
    --input /path/to/test/roi_data \
    --output ./predictions \
    --models_dir ./checkpoints/task2_resnet18_baseline \
    --group 1
```

#### Key Arguments for Evaluation

| Argument | Description |
|---|---|
| `--config` | Configuration file with model settings (optional, but recommended) |
| `--input` | Path to directory containing test ROI files |
| `--output` | Directory where `predictions.csv` will be saved |
| `--models_dir` | Directory containing `cv_results.json` (ensemble) or `best_model.pth` (single model) |
| `--model` | Explicit path to single `.pth` checkpoint (overrides `--models_dir`) |
| `--model_type` | Model architecture: `resnet18`, `resnet50`, `densenet121` (required if `--config` not provided) |
| `--num_classes` | Number of output classes (required if `--config` not provided) |
| `--classes` | Class names in label-index order (required if `--config` not provided) |
| `--device` | Device: `cuda` or `cpu` (auto-detected if omitted) |
| `--group` | Group number for output filename (e.g., 1 → `group1_task2_results.csv`) |

---

## Alternative: 2D Dataset Approach

Instead of using 3D models, you can extract 2D slices from the ROI volumes and train 2D CNN models. This approach reduces memory requirements and allows leveraging pre-trained 2D models.

### Extract 2D Slices

```bash
cd task2_classification

# Training data with labels
python prepare_2d_dataset_for_task2.py \
    --input_path /path/to/roi_data \
    --output_path /path/to/2d_dataset \
    --labels_csv /path/to/labels.csv \
    --slice_strategy "middle_n" \
    --num_slices 5 \
    --num_workers 8
```

**Output structure:**
```
2d_dataset/
├── patient_001/
│   ├── 000.png
│   ├── 001.png
│   ├── 002.png
│   └── 003.png
├── patient_002/
│   ├── 000.png
│   ├── 001.png
│   └── ...
└── ...
```

### Slice Selection Strategies

All strategies automatically exclude empty slices and only include them if insufficient non-empty slices exist to meet the target count.

| Strategy | Description | Use Case |
|---|---|---|
| `all_nonempty` | All non-empty slices | Maximum meaningful data |
| `middle_n` | Middle N non-empty slices centered around volume | Balanced, captures tumor center |
| `center_single` | Single center non-empty slice | Fastest, minimal information |
| `equidistant` | N equally spaced non-empty slices | Balanced distribution across volume |

### Prepare Test Data

```bash
python prepare_2d_dataset_for_task2.py \
    --input_path /path/to/test_roi_data \
    --output_path /path/to/2d_test_dataset \
    --test-mode \
    --slice_strategy "middle_n" \
    --num_slices 5
```

### Key Arguments for 2D Dataset Preparation

| Argument | Description |
|---|---|
| `--input_path` | Directory containing NIfTI ROI files or `roi_data/` subdirectory |
| `--output_path` | Output directory (organized by patient_id/) |
| `--labels_csv` | CSV with `patient_id` and `type` columns (required for training) |
| `--test-mode` | Process test data without labels |
| `--slice_strategy` | Strategy for slice selection: `all_nonempty`, `middle_n`, `center_single`, `equidistant` |
| `--num_slices` | Number of slices for `middle_n` and `equidistant` strategies (default: 5) |
| `--window_center` | HU window center (default: 40 for abdomen soft-tissue) |
| `--window_width` | HU window width (default: 400, clips to [−160, 240]) |
| `--num_workers` | Number of parallel worker processes |

### Next Steps

Once slices are extracted:
1. Train 2D models (ResNet, DenseNet, etc.) on the extracted slices
2. Aggregate predictions across slices (average, majority vote, or attention-based)
3. Generate final patient-level predictions

---

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
│   └── model_checkpoints/             # Saved models (generated)
|
├── task2_classification/
│   ├── train.py                       # Training with k-fold CV
│   ├── evaluate.py                    # Evaluation script
│   ├── prepare_2d_dataset_for_task2.py # Extract 2D slices for 2D models
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
|   └── model_checkpoints/             # Saved models (generated)
```

---

## 📤 Submission

### Workflow Overview

**Tasks 1 and 2 are completely independent** — you can work on either or both in parallel. No dependencies between tasks:

```
Training Phase (Fully Parallel):
├─ Train Task 1 model (see Task 1: Quick Start)
└─ Train Task 2 model (see Task 2: Quick Start)

Evaluation & Submission Phase (Fully Parallel):
├─ Evaluate Task 1 on test data → generates predictions
│  └─ Submit: group<N>_task1_results.zip
└─ Evaluate Task 2 on test data → generates predictions
   └─ Submit: group<N>_task2_results.csv
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
- [ ] Task 2: Model trained and checkpoint saved
- [ ] Task 2: Predictions generated for test set
- [ ] Task 2: `group<N>_task2_results.csv` created
- [ ] Both tasks: Results in correct format and ready for submission
---

**Good luck! 🚀**  
