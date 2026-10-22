"""
Script 2: Prepare Dataset (scripts/prepare_dataset.py)
Creates deterministic, reproducible, stratified splits (70% Train, 15% Val, 15% Test)
and persists split manifests with zero data leakage.
"""

import argparse
import sys
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_DATA_DIR,
    SPLITS_DIR,
    SPLIT_MANIFEST_PATH,
    RANDOM_SEED,
)
from src.preprocessing import create_stratified_manifest
from src.reproducibility import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Create deterministic stratified train/val/test splits.")
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        default=[str(PROJECT_ROOT / "train"), str(PROJECT_ROOT / "test"), str(RAW_DATA_DIR)],
        help="Directories to search for images",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(SPLIT_MANIFEST_PATH),
        help="Path to save output manifest JSON",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for splitting")
    args = parser.parse_args()

    seed_everything(args.seed)

    data_dirs = [Path(d) for d in args.data_dirs if Path(d).exists()]
    if not data_dirs:
        print(f"Error: None of the specified directories exist: {args.data_dirs}")
        sys.exit(1)

    print(f"Preparing dataset splits from directories: {[str(d) for d in data_dirs]}")
    manifest = create_stratified_manifest(
        data_dirs=data_dirs,
        output_path=Path(args.output_path),
        seed=args.seed,
    )

    meta = manifest["metadata"]
    print("\nDataset Partitioning Summary:")
    print(f"  Total Valid Samples: {meta['total_samples']}")
    print(f"  Train Samples (70%): {meta['train_samples']}")
    print(f"  Val Samples   (15%): {meta['val_samples']}")
    print(f"  Test Samples  (15%): {meta['test_samples']}")
    print(f"Manifest written to: {args.output_path}")


if __name__ == "__main__":
    main()
