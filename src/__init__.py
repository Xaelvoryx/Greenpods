"""
Avocado Ripening Stage Detection & Classification Package.
Export key components for seamless imports across scripts, notebooks, and tests.
"""

from src.config import (
    CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SPLITS_DIR,
    RANDOM_SEED,
    DEVICE,
)

__version__ = "1.0.0"
__all__ = [
    "CLASSES",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "PROJECT_ROOT",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "SPLITS_DIR",
    "RANDOM_SEED",
    "DEVICE",
]
