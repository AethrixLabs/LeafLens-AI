# LeafLens-AI

AI-powered crop disease detection and decision-support system designed to assist farmers in identifying crop diseases accurately and understanding the reliability of those predictions.

## 🚀 Project Vision
LeafLens-AI combines computer vision, explainable AI, and human feedback to move beyond black-box predictions and build trustable agricultural AI. The goal is to not only detect diseases from leaf images but also to explain predictions in human-understandable ways and provide confidence-aware insights to support better decision-making on the field.

## 🎯 Core Objectives
- Accurately detect crop diseases from leaf images using crop-specific models
- Provide visual and textual explanations (explainable AI) to make predictions interpretable
- Collect farmer feedback to enable human-in-the-loop improvements
- Produce a trust score for each prediction to communicate reliability
- Support farmer-friendly local language outputs and offline-aware workflows

## 🌾 Crops Covered (Current Scope)
LeafLens-AI currently supports four major food crops (each handled independently):
- Rice
- Wheat
- Potato
- Maize (Corn)

## 🧩 Core Features
1. Image-Based Crop Disease Detection
   - Farmers provide a leaf image; a crop-specific CNN predicts the disease.
   - Models are trained with curated datasets for each crop.

2. Explainable AI Diagnosis
   - Highlights infected regions on the leaf (visual explanation via Grad-CAM).
   - Provides a short textual reason for the prediction to help non-technical users.

3. Farmer Validation & Feedback Learning
   - Farmers can mark predictions as Correct / Incorrect.
   - Feedback is stored for continual improvement (human-in-the-loop learning).

4. Trust Score System
   - Each prediction receives a trust level based on model confidence, prediction consistency, and farmer feedback.
   - Prevents over-reliance on uncertain predictions and surfaces low-confidence cases.

5. Local Language Support
   - Results (disease name, explanation, and trust messages) can be presented in farmer-friendly local languages and simple wording.

6. Offline-Aware Design (Logic Level)
   - Supports image capture and queuing; performs predictions when connectivity is available.
   - Designed for rural constraints and intermittent connectivity.

7. Context-Aware Advisory (Weather & Market)
   - Optionally combines predictions with weather data and market (mandi) information to provide risk-aware suggestions.

## 🏗️ System Architecture (High-Level)
- User selects the crop and uploads an image
- A shared preprocessing pipeline prepares the image (`preprocessing/`)
- The request is routed to a crop-specific model defined in `models/`
- Explainable AI (`explainability/`) generates visual and textual reasoning
- Trust score and optional advisory modules add confidence and actionable context

## 📁 Repository Structure (high level)
- `app/` - entry points and API (`main.py`, `api.py`)
- `training/` - training scripts per crop (`train_*.py`)
- `models/` - model definitions per crop
- `Dataset_Crop/` - dataset folders for training/validation
- `explainability/` - `gradcam.py`, visualization helpers
- `inference/` - `predictor.py`, `router.py` for inference logic
- `trust_score/` - `trust_logic.py` for trust computations
- `preprocessing/`, `utils/`, `config/` - supporting utilities and configs

## ⚡ Quick Start
1. Create and activate a virtual environment
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run API (local):

```bash
python app/main.py
```

4. Run a training script (example):

```bash
# Basic training with default config
python training/train_corn.py

# Compute dataset statistics first
python training/train_corn.py --compute-stats

# Use computed statistics instead of ImageNet defaults
python training/train_corn.py --compute-stats --use-dataset-stats
```

## 🗂️ Dataset Layout
Place labeled images under `Dataset_Crop/{Train,Val}/{Crop}/{Class}/`. See existing folders for examples (Rice, Wheat, Potato, Corn).

## 🔧 Improved Data Loading & Training Features

The preprocessing pipeline has been enhanced with production-ready features for better experimentation and reproducibility:

### Configurable Augmentation
- **Config-driven**: Augmentation parameters moved to `config/config.yaml`
- **Easy toggling**: Enable/disable augmentation without code changes
- **Domain-aware**: Removed vertical flip (leaves don't appear upside-down)
- **RGB enforcement**: Automatic conversion to RGB to prevent channel mismatches

### Reproducibility & Performance
- **Deterministic mode**: `set_seed()` function for reproducible experiments
- **Persistent workers**: Faster training with `persistent_workers=True`
- **Class imbalance detection**: Automatic logging of class distributions
- **Dataset statistics**: Optional computation of custom mean/std from your data

### Test-Time Augmentation (TTA)
- **Multiple transforms**: Apply different augmentations at inference time
- **Improved accuracy**: Average predictions across augmented versions
- **Configurable**: Enable/disable and set number of augmentations

### Usage Examples

**Basic training with config:**
```python
from preprocessing.dataset_loader import create_dataloaders_from_config, set_seed

# Set seed for reproducibility
set_seed(42)

# Create dataloaders from config
train_loader, val_loader, idx_to_class = create_dataloaders_from_config(
    "Dataset_Crop/Train/Corn", 
    "Dataset_Crop/Val/Corn"
)
```

**Compute custom dataset statistics:**
```python
from preprocessing.dataset_loader import create_image_datasets, compute_dataset_stats

_, val_dataset = create_image_datasets(train_dir, val_dir, augment=False)
mean, std = compute_dataset_stats(val_dataset)
# Use these instead of ImageNet stats for potentially better performance
```

**Test-time augmentation:**
```python
from preprocessing.image_transforms import get_tta_transforms

tta_transforms = get_tta_transforms(input_size=224)
# Apply each transform and average predictions
```

## 🧪 Explainability & Trust
- Visual explanations: `explainability/gradcam.py`
- Trust logic: `trust_score/trust_logic.py` — combines model confidence and feedback

## 🤝 Contributing
Contributions are welcome — please open issues or pull requests. Add new crops following the structure in `training/`, `models/`, and `Dataset_Crop/`.

## 📄 License
This project is licensed under the MIT License. See `LICENSE` for details.

---

If you'd like, I can also add example commands, CI steps, or expand sections (dataset prep, evaluation metrics, or deployment instructions).