"""
Script 1: Audit Dataset (scripts/audit_dataset.py)
Discovers all dataset files, validates readability, checks file integrity,
records per-class distribution, and plots representative sample grids and charts.
"""

import argparse
import sys
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_DATA_DIR,
    EDA_OUTPUTS_DIR,
    CLASSES,
)
from src.data_audit import run_full_dataset_audit, discover_image_files
from src.visualization import plot_class_distribution, plot_representative_samples


def main():
    parser = argparse.ArgumentParser(description="Audit and validate dataset quality and distribution.")
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        default=[str(PROJECT_ROOT / "train"), str(PROJECT_ROOT / "test"), str(RAW_DATA_DIR)],
        help="Directories to search for images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(EDA_OUTPUTS_DIR),
        help="Directory to save audit summaries and figures",
    )
    args = parser.parse_args()

    data_dirs = [Path(d) for d in args.data_dirs if Path(d).exists()]
    if not data_dirs:
        print(f"Error: None of the specified directories exist: {args.data_dirs}")
        sys.exit(1)

    print(f"Running dataset audit on directories: {[str(d) for d in data_dirs]}...")
    out_dir = Path(args.output_dir)
    audit_summary = run_full_dataset_audit(data_dirs, output_dir=out_dir)

    print(f"\nAudit Summary:")
    print(f"  Total Images Discovered: {audit_summary.get('total_images', 0)}")
    print(f"  Valid Images:            {audit_summary.get('valid_images', 0)}")
    print(f"  Corrupted Images:        {audit_summary.get('corrupted_images', 0)}")
    print(f"  Duplicate Files (MD5):   {audit_summary.get('duplicate_count', 0)}")
    print(f"  Class Distribution:      {audit_summary.get('class_distribution', {})}")

    # Generate visual artifacts
    plot_class_distribution(
        class_dist_csv=out_dir / "class_distribution.csv",
        save_path=out_dir / "class_distribution.png",
    )

    # Gather representative samples for plotting
    samples_per_class = {c: [] for c in CLASSES}
    records = []
    for d in data_dirs:
        records.extend(discover_image_files(d))
    for r in records:
        c = r["class_name"]
        if c in samples_per_class and len(samples_per_class[c]) < 4:
            samples_per_class[c].append(r["file_path"])

    plot_representative_samples(samples_per_class, save_path=out_dir / "representative_samples.png")
    print(f"All EDA artifacts generated successfully in {out_dir}")


if __name__ == "__main__":
    main()
