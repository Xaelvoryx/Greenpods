"""
Unit and integration tests for preprocessing and deterministic stratified splitting.
"""

from pathlib import Path
import pytest
from PIL import Image
import pandas as pd

from src.config import CLASSES, CLASS_TO_IDX
from src.preprocessing import create_stratified_manifest, load_split_manifest


def test_stratified_split_zero_leakage(tmp_path):
    """Test creating stratified splits with zero partition overlap."""
    dataset_dir = tmp_path / "raw"
    # Create synthetic structure with 20 images per class
    for c in CLASSES:
        cdir = dataset_dir / c
        cdir.mkdir(parents=True)
        for i in range(20):
            img = Image.new("RGB", (64, 64), color=(i * 10, i * 5, 50))
            img.save(cdir / f"img_{i}.png")

    manifest_path = tmp_path / "manifest.json"
    manifest = create_stratified_manifest(
        data_dirs=[dataset_dir],
        output_path=manifest_path,
        seed=42,
    )

    assert manifest_path.exists()
    assert manifest["metadata"]["total_samples"] == 80
    assert manifest["metadata"]["train_samples"] == int(80 * 0.70)

    # Check zero overlap
    df = load_split_manifest(manifest_path)
    train_paths = set(df[df["split"] == "train"]["file_path"])
    val_paths = set(df[df["split"] == "val"]["file_path"])
    test_paths = set(df[df["split"] == "test"]["file_path"])

    assert len(train_paths.intersection(val_paths)) == 0
    assert len(train_paths.intersection(test_paths)) == 0
    assert len(val_paths.intersection(test_paths)) == 0
