"""
Dataset discovery, auditing, and health checking module.
Scans image directories, computes image dimensions, aspect ratios, color modes,
and detects corrupted files, extreme outliers, or class imbalances.
"""

from pathlib import Path
import os
import hashlib
import json
from typing import Dict, List, Tuple, Any
from PIL import Image
import pandas as pd

from src.config import (
    CLASSES,
    CLASS_TO_IDX,
    FOLDER_TO_CLASS_MAP,
    SUPPORTED_EXTENSIONS,
    RAW_DATA_DIR,
    EDA_OUTPUTS_DIR,
)


def discover_image_files(base_dir: Path) -> List[Dict[str, Any]]:
    """
    Recursively scan directories to find all supported images and resolve their class.
    Supports canonical class subfolders (underripe, breaking, ripe, overripe)
    and mapped subfolders (e.g., train/grenning -> underripe).
    """
    records = []
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return records

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            file_path = Path(root) / file
            ext = file_path.suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                # Resolve class from immediate parent or path parts
                parent_name = file_path.parent.name
                resolved_class = FOLDER_TO_CLASS_MAP.get(parent_name, None)
                if resolved_class is None:
                    # Check higher directory parts
                    for part in reversed(file_path.parts[:-1]):
                        if part in FOLDER_TO_CLASS_MAP:
                            resolved_class = FOLDER_TO_CLASS_MAP[part]
                            break

                records.append({
                    "file_path": str(file_path.resolve()),
                    "relative_path": str(file_path),
                    "file_name": file,
                    "extension": ext,
                    "folder_name": parent_name,
                    "class_name": resolved_class if resolved_class else "unknown",
                    "file_size_bytes": file_path.stat().st_size,
                })
    return records


def audit_image_file(file_path: str) -> Dict[str, Any]:
    """Inspect an individual image file for readability, dimensions, and color mode."""
    res = {
        "is_valid": False,
        "width": 0,
        "height": 0,
        "channels": 0,
        "mode": "",
        "aspect_ratio": 0.0,
        "error": None,
        "md5": None,
    }
    p = Path(file_path)
    try:
        # Compute md5 for duplicate detection
        with open(p, "rb") as f:
            file_bytes = f.read()
            res["md5"] = hashlib.md5(file_bytes).hexdigest()

        with Image.open(p) as img:
            res["width"], res["height"] = img.size
            res["mode"] = img.mode
            res["channels"] = len(img.getbands())
            res["aspect_ratio"] = round(res["width"] / max(res["height"], 1), 3)
            res["is_valid"] = True
    except Exception as e:
        res["error"] = str(e)
    return res


def run_full_dataset_audit(data_dirs: List[Path], output_dir: Path = EDA_OUTPUTS_DIR) -> Dict[str, Any]:
    """Execute complete dataset audit and generate summary tables and JSON metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for d in data_dirs:
        records = discover_image_files(d)
        all_records.extend(records)

    if not all_records:
        summary = {"error": "No image files found in specified directories.", "total_images": 0}
        with open(output_dir / "dataset_audit.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    df = pd.DataFrame(all_records)
    print(f"Auditing {len(df)} images across directories...")

    audit_details = []
    for path in df["file_path"]:
        audit_details.append(audit_image_file(path))

    audit_df = pd.DataFrame(audit_details)
    full_df = pd.concat([df, audit_df], axis=1)

    # Calculate statistics
    total_images = len(full_df)
    valid_images = full_df["is_valid"].sum()
    corrupted_images = total_images - valid_images
    duplicate_count = full_df.duplicated(subset=["md5"]).sum()

    # Per-class distribution
    class_counts = full_df[full_df["class_name"].isin(CLASSES)]["class_name"].value_counts().to_dict()
    # Ensure all canonical classes are represented
    for c in CLASSES:
        if c not in class_counts:
            class_counts[c] = 0

    class_dist_df = pd.DataFrame([
        {
            "class_name": c,
            "count": class_counts[c],
            "percentage": round(class_counts[c] * 100.0 / max(total_images, 1), 2)
        }
        for c in CLASSES
    ])

    # Quality summaries
    quality_summary_df = pd.DataFrame([
        {"metric": "Total Images Discovered", "value": total_images},
        {"metric": "Valid Readable Images", "value": valid_images},
        {"metric": "Corrupted / Unreadable Images", "value": corrupted_images},
        {"metric": "Exact Duplicate Images (MD5)", "value": duplicate_count},
        {"metric": "Mean Width", "value": round(full_df[full_df["is_valid"]]["width"].mean(), 1)},
        {"metric": "Mean Height", "value": round(full_df[full_df["is_valid"]]["height"].mean(), 1)},
        {"metric": "Min File Size (KB)", "value": round(full_df["file_size_bytes"].min() / 1024, 1)},
        {"metric": "Max File Size (KB)", "value": round(full_df["file_size_bytes"].max() / 1024, 1)},
    ])

    # Export CSVs and JSON
    class_dist_df.to_csv(output_dir / "class_distribution.csv", index=False)
    quality_summary_df.to_csv(output_dir / "image_quality_summary.csv", index=False)
    full_df.to_csv(output_dir / "full_image_audit_records.csv", index=False)

    audit_summary = {
        "total_images": int(total_images),
        "valid_images": int(valid_images),
        "corrupted_images": int(corrupted_images),
        "duplicate_count": int(duplicate_count),
        "class_distribution": {k: int(v) for k, v in class_counts.items()},
        "formats": full_df["extension"].value_counts().to_dict(),
        "color_modes": full_df["mode"].value_counts().to_dict(),
    }

    with open(output_dir / "dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print(f"Dataset audit complete. Audit artifacts saved in {output_dir}")
    return audit_summary
