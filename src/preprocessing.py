"""
Data preprocessing, manifest creation, and deterministic stratified splitting.
Enforces zero data leakage: 70% Train, 15% Validation, 15% Test.
Manifests saved with filepaths, canonical labels, split tags, and file hashes.
"""

from pathlib import Path
import json
import random
from typing import Dict, List, Tuple, Any
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CLASSES,
    CLASS_TO_IDX,
    SPLITS_DIR,
    SPLIT_MANIFEST_PATH,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
)
from src.data_audit import discover_image_files, audit_image_file


def create_stratified_manifest(
    data_dirs: List[Path],
    output_path: Path = SPLIT_MANIFEST_PATH,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Scan image files, filter valid images, perform deterministic stratified splitting (70/15/15),
    and persist split manifest JSON.
    """
    records = []
    for d in data_dirs:
        records.extend(discover_image_files(d))

    if not records:
        raise ValueError("No images discovered in provided data directories.")

    # Filter to known canonical classes
    valid_records = [r for r in records if r["class_name"] in CLASSES]
    if not valid_records:
        raise ValueError(f"No records mapped to canonical classes {CLASSES}.")

    df = pd.DataFrame(valid_records)
    print(f"Creating stratified splits for {len(df)} images across {len(CLASSES)} classes...")

    # First split: train vs (val + test)
    val_test_ratio = VAL_RATIO + TEST_RATIO
    train_df, val_test_df = train_test_split(
        df,
        test_size=val_test_ratio,
        random_state=seed,
        stratify=df["class_name"],
    )

    # Second split: val vs test (equal split of the remaining 30% -> 15% and 15%)
    relative_test_ratio = TEST_RATIO / val_test_ratio
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=val_test_df["class_name"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined_df["class_idx"] = combined_df["class_name"].map(CLASS_TO_IDX)

    # Verify zero overlap between partition file paths
    train_paths = set(train_df["file_path"])
    val_paths = set(val_df["file_path"])
    test_paths = set(test_df["file_path"])

    assert len(train_paths.intersection(val_paths)) == 0, "Data leakage detected: train and val overlap!"
    assert len(train_paths.intersection(test_paths)) == 0, "Data leakage detected: train and test overlap!"
    assert len(val_paths.intersection(test_paths)) == 0, "Data leakage detected: val and test overlap!"

    # Create manifest dictionary
    manifest = {
        "metadata": {
            "total_samples": len(combined_df),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "random_seed": seed,
            "classes": CLASSES,
            "proportions": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        },
        "samples": combined_df.to_dict(orient="records"),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Also save convenient CSV manifests
    train_df.to_csv(output_path.parent / "train_manifest.csv", index=False)
    val_df.to_csv(output_path.parent / "val_manifest.csv", index=False)
    test_df.to_csv(output_path.parent / "test_manifest.csv", index=False)

    print(f"Manifest created successfully: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test.")
    return manifest


def load_split_manifest(manifest_path: Path = SPLIT_MANIFEST_PATH) -> pd.DataFrame:
    """Load split manifest into a pandas DataFrame."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found at {manifest_path}. Run prepare_dataset.py first.")
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data["samples"])
