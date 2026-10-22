"""
Unified evaluation module for assessing Classical Machine Learning and Deep Learning models
on identical held-out test splits. Computes accuracy, macro precision, macro recall,
macro F1-score, per-class metrics, confusion matrices, and inference latency.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import time
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from PIL import Image

from src.config import (
    CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    DEVICE,
    EVALUATION_OUTPUTS_DIR,
    CLASSICAL_MODEL_PATH,
    DL_MODEL_PATH,
)
from src.classical_model import load_classical_model, extract_classical_features
from src.dl_model import load_dl_model
from src.dataset import get_transforms
from src.visualization import plot_confusion_matrix, plot_model_comparison


def evaluate_classical(
    test_df: pd.DataFrame,
    model_path: Path = CLASSICAL_MODEL_PATH,
    output_dir: Path = EVALUATION_OUTPUTS_DIR,
) -> Tuple[Dict[str, Any], np.ndarray, float]:
    """Evaluate trained Classical ML pipeline on held-out test set."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classical_artifact = load_classical_model(model_path)
    pipeline = classical_artifact["pipeline"]

    print(f"Extracting features and evaluating classical model on {len(test_df)} test samples...")
    X_test = []
    y_test = []
    sample_paths = []

    for _, row in test_df.iterrows():
        try:
            feats = extract_classical_features(row["file_path"])
            X_test.append(feats)
            y_test.append(row["class_idx"])
            sample_paths.append(row["file_path"])
        except Exception as e:
            print(f"Skipping unreadable sample {row['file_path']}: {e}")

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Benchmark inference latency per image (warmup + multiple runs)
    dummy_input = X_test[0:1]
    for _ in range(5):
        _ = pipeline.predict(dummy_input)

    latencies = []
    for i in range(min(50, len(X_test))):
        t0 = time.perf_counter()
        _ = pipeline.predict(X_test[i:i+1])
        latencies.append((time.perf_counter() - t0) * 1000.0)
    avg_latency_ms = float(np.mean(latencies))

    y_pred = pipeline.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        target_names=CLASSES,
        labels=[CLASS_TO_IDX[c] for c in CLASSES],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[CLASS_TO_IDX[c] for c in CLASSES],
    )

    # Plot and save confusion matrix
    plot_confusion_matrix(
        cm,
        class_names=CLASSES,
        save_path=output_dir / "confusion_matrix_classical.png",
        title="Classical ML (SVM) - Test Confusion Matrix",
    )

    # Save detailed classification report
    with open(output_dir / "classification_report_classical.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report, cm, avg_latency_ms


def evaluate_deep_learning(
    test_df: pd.DataFrame,
    checkpoint_path: Path = DL_MODEL_PATH,
    output_dir: Path = EVALUATION_OUTPUTS_DIR,
    device: str = DEVICE,
) -> Tuple[Dict[str, Any], np.ndarray, float]:
    """Evaluate trained MobileNetV2 checkpoint on held-out test set."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_dl_model(checkpoint_path, device=device)
    val_transform = get_transforms(split="val")

    y_true = []
    y_pred = []
    sample_paths = []

    print(f"Evaluating MobileNetV2 on {len(test_df)} test samples ({device})...")
    with torch.no_grad():
        for _, row in test_df.iterrows():
            img_path = row["file_path"]
            try:
                img = Image.open(img_path).convert("RGB")
                tensor = val_transform(img).unsqueeze(0).to(device)
                output = model(tensor)
                pred_idx = torch.argmax(output, dim=1).item()
                y_true.append(row["class_idx"])
                y_pred.append(pred_idx)
                sample_paths.append(img_path)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    # Latency measurement on single image tensor
    dummy_tensor = torch.randn(1, 3, 224, 224).to(device)
    for _ in range(10):
        _ = model(dummy_tensor)
    if device == "cuda":
        torch.cuda.synchronize()

    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = model(dummy_tensor)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    avg_latency_ms = float(np.mean(latencies))

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        labels=[CLASS_TO_IDX[c] for c in CLASSES],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[CLASS_TO_IDX[c] for c in CLASSES],
    )

    plot_confusion_matrix(
        cm,
        class_names=CLASSES,
        save_path=output_dir / "confusion_matrix_mobilenetv2.png",
        title="MobileNetV2 - Test Confusion Matrix",
    )

    with open(output_dir / "classification_report_mobilenetv2.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report, cm, avg_latency_ms


def generate_model_comparison(
    classical_report: Dict[str, Any],
    dl_report: Dict[str, Any],
    classical_latency_ms: float,
    dl_latency_ms: float,
    output_dir: Path = EVALUATION_OUTPUTS_DIR,
) -> pd.DataFrame:
    """Generate comparative summary table and plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate model sizes
    classical_size_mb = (
        CLASSICAL_MODEL_PATH.stat().st_size / (1024 * 1024)
        if CLASSICAL_MODEL_PATH.exists()
        else 0.0
    )
    dl_size_mb = (
        DL_MODEL_PATH.stat().st_size / (1024 * 1024)
        if DL_MODEL_PATH.exists()
        else 0.0
    )

    comparison_data = [
        {"Metric": "Accuracy", "Classical ML": round(classical_report["accuracy"], 4), "MobileNetV2": round(dl_report["accuracy"], 4)},
        {"Metric": "Macro Precision", "Classical ML": round(classical_report["macro avg"]["precision"], 4), "MobileNetV2": round(dl_report["macro avg"]["precision"], 4)},
        {"Metric": "Macro Recall", "Classical ML": round(classical_report["macro avg"]["recall"], 4), "MobileNetV2": round(dl_report["macro avg"]["recall"], 4)},
        {"Metric": "Macro F1-score", "Classical ML": round(classical_report["macro avg"]["f1-score"], 4), "MobileNetV2": round(dl_report["macro avg"]["f1-score"], 4)},
        {"Metric": "CPU Inference Latency (ms)", "Classical ML": round(classical_latency_ms, 2), "MobileNetV2": round(dl_latency_ms, 2)},
        {"Metric": "Model Size (MB)", "Classical ML": round(classical_size_mb, 2), "MobileNetV2": round(dl_size_mb, 2)},
    ]

    df = pd.DataFrame(comparison_data)
    df.to_csv(output_dir / "model_comparison.csv", index=False)

    with open(output_dir / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)

    plot_model_comparison(df, save_path=output_dir / "model_comparison.png")
    print(f"Model comparison table and figure generated at {output_dir}")
    return df
