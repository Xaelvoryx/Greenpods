"""
Visualization utilities for Exploratory Data Analysis, color channel histograms,
representative sample grids, confusion matrices, and training curves.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image

from src.config import CLASSES, EDA_OUTPUTS_DIR, EVALUATION_OUTPUTS_DIR, EXPERIMENTS_OUTPUTS_DIR


def plot_class_distribution(
    class_dist_csv: Path = EDA_OUTPUTS_DIR / "class_distribution.csv",
    save_path: Path = EDA_OUTPUTS_DIR / "class_distribution.png",
) -> None:
    """Plot clear bar chart of per-class distribution."""
    if not class_dist_csv.exists():
        return
    df = pd.read_csv(class_dist_csv)
    plt.figure(figsize=(8, 4.5))
    palette = ["#2d6a4f", "#74c69d", "#d8572a", "#3c096c"]
    bars = plt.bar(df["class_name"], df["count"], color=palette[:len(df)], edgecolor="black", alpha=0.9)
    plt.title("Avocado Ripening Stage Class Distribution", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Ripening Stage", fontsize=12)
    plt.ylabel("Sample Count", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 5, f"{int(h)}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_representative_samples(
    samples_per_class: Dict[str, List[str]],
    save_path: Path = EDA_OUTPUTS_DIR / "representative_samples.png",
) -> None:
    """Plot representative grid of avocado images across all 4 ripening stages."""
    num_classes = len(CLASSES)
    num_cols = max(len(v) for v in samples_per_class.values()) if samples_per_class else 4
    num_cols = min(num_cols, 4)

    fig, axes = plt.subplots(num_classes, num_cols, figsize=(num_cols * 3, num_classes * 3))
    if num_classes == 1:
        axes = np.expand_dims(axes, 0)
    if num_cols == 1:
        axes = np.expand_dims(axes, 1)

    for r_idx, cls_name in enumerate(CLASSES):
        img_paths = samples_per_class.get(cls_name, [])
        for c_idx in range(num_cols):
            ax = axes[r_idx, c_idx]
            if c_idx < len(img_paths) and Path(img_paths[c_idx]).exists():
                try:
                    img = Image.open(img_paths[c_idx]).convert("RGB")
                    ax.imshow(img)
                    if c_idx == 0:
                        ax.set_ylabel(cls_name.upper(), fontsize=11, fontweight="bold")
                    ax.set_title(Path(img_paths[c_idx]).name, fontsize=8)
                except Exception:
                    ax.text(0.5, 0.5, "Error", ha="center", va="center")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("Representative Avocado Ripening Stage Samples", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save standardized confusion matrix heatmap."""
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
    )
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Stage", fontsize=11)
    plt.ylabel("Ground Truth Stage", fontsize=11)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Path = EXPERIMENTS_OUTPUTS_DIR / "training_curves.png",
) -> None:
    """Plot loss and accuracy curves across training epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss plot
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1d3557")
    ax1.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#e63946")
    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train Acc (%)", color="#2a9d8f")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "s--", label="Val Acc (%)", color="#e76f51")
    ax2.set_title("Classification Accuracy vs. Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_model_comparison(
    metrics_df: pd.DataFrame,
    save_path: Path = EVALUATION_OUTPUTS_DIR / "model_comparison.png",
) -> None:
    """Generate comparative grouped bar chart for Classical ML vs. Deep Learning."""
    plt.figure(figsize=(9, 5))
    metrics = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-score"]
    
    # Filter for standard metrics
    plot_df = metrics_df[metrics_df["Metric"].isin(metrics)].copy()
    
    x = np.arange(len(metrics))
    width = 0.35

    classical_vals = [
        float(plot_df[plot_df["Metric"] == m]["Classical ML"].values[0])
        for m in metrics
    ]
    dl_vals = [
        float(plot_df[plot_df["Metric"] == m]["MobileNetV2"].values[0])
        for m in metrics
    ]

    plt.bar(x - width/2, classical_vals, width, label="Classical ML (SVM)", color="#457b9d", edgecolor="black")
    plt.bar(x + width/2, dl_vals, width, label="Deep Learning (MobileNetV2)", color="#2a9d8f", edgecolor="black")

    plt.title("Model Performance Comparison on Held-Out Test Set", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Evaluation Metric", fontsize=11)
    plt.ylabel("Score (0.0 - 1.0)", fontsize=11)
    plt.xticks(x, metrics)
    plt.ylim(0, 1.1)
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
