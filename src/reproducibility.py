"""
Reproducibility utilities for setting deterministic random seeds and recording environment metadata.
"""

import os
import random
import sys
import platform
import json
from pathlib import Path
import numpy as np
import torch

from src.config import RANDOM_SEED


def seed_everything(seed: int = RANDOM_SEED) -> None:
    """Set seeds across Python random, NumPy, PyTorch CPU and GPU for deterministic runs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_environment_info() -> dict:
    """Capture full runtime environment metadata for scientific auditability."""
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "os": platform.system(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "numpy_version": np.__version__,
    }
    return info


def save_environment_metadata(output_path: Path) -> None:
    """Save environment info to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    info = get_environment_info()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
