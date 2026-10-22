"""
Configuration module for the Avocado Ripening Stage Detection & Classification System.
Defines paths, class mappings, hyperparameters, and reproducibility constants.
"""

from pathlib import Path
import os
import torch

# Base project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
CLASSICAL_MODELS_DIR = MODELS_DIR / "classical"
DL_MODELS_DIR = MODELS_DIR / "deep_learning"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EDA_OUTPUTS_DIR = OUTPUTS_DIR / "eda"
EVALUATION_OUTPUTS_DIR = OUTPUTS_DIR / "evaluation"
EXPERIMENTS_OUTPUTS_DIR = OUTPUTS_DIR / "experiments"
PREDICTIONS_OUTPUTS_DIR = OUTPUTS_DIR / "predictions"
DOCS_DIR = PROJECT_ROOT / "docs"
REPORT_DIR = PROJECT_ROOT / "report"

# Canonical 4-class Ripening Stage definition
CLASSES = ["underripe", "breaking", "ripe", "overripe"]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(CLASSES)}

# Workspace extracted folder mappings / aliases
FOLDER_TO_CLASS_MAP = {
    # Direct canonical classes
    "underripe": "underripe",
    "breaking": "breaking",
    "ripe": "ripe",
    "overripe": "overripe",
    # Case variations
    "Underripe": "underripe",
    "Breaking": "breaking",
    "Ripe": "ripe",
    "Overripe": "overripe",
    # Workspace dataset folder mappings (ensuring 100% full dataset utilization)
    "grenning": "underripe",
    "fresh": "breaking",
    "canker": "ripe",
    "blackspot": "overripe"
}

# Image Processing Specifications
DL_IMAGE_SIZE = (224, 224)
CLASSICAL_IMAGE_SIZE = (128, 128)
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# Dataset Splitting Proportions
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Normalization constants (ImageNet defaults)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Hardware & Training Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32 if torch.cuda.is_available() else 16
NUM_WORKERS = 0  # Safe default on Windows
LEARNING_RATE = 1e-3
NUM_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 4

# Model Artifact Locations
CLASSICAL_MODEL_PATH = CLASSICAL_MODELS_DIR / "avocado_classical_model.joblib"
DL_MODEL_PATH = DL_MODELS_DIR / "avocado_mobilenetv2_best.pth"
SPLIT_MANIFEST_PATH = SPLITS_DIR / "dataset_manifest.json"
