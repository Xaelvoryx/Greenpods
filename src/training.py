"""
Training execution loop for MobileNetV2 with Adam optimizer, CrossEntropyLoss,
validation tracking, and early stopping.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    DEVICE,
    LEARNING_RATE,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    DL_MODEL_PATH,
    EXPERIMENTS_OUTPUTS_DIR,
)
from src.dl_model import build_avocado_classifier, save_checkpoint
from src.visualization import plot_training_curves


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = DEVICE,
) -> Tuple[float, float]:
    """Train for one epoch across batches."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels, _ in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = DEVICE,
) -> Tuple[float, float]:
    """Evaluate model on validation or test split."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / max(total, 1)
    val_acc = correct / max(total, 1)
    return val_loss, val_acc


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    patience: int = EARLY_STOPPING_PATIENCE,
    device: str = DEVICE,
    checkpoint_path: Path = DL_MODEL_PATH,
    experiments_dir: Path = EXPERIMENTS_OUTPUTS_DIR,
) -> Dict[str, Any]:
    """Complete deep learning training workflow with early stopping and history logging."""
    experiments_dir = Path(experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    print(f"Initializing MobileNetV2 on device: {device}")
    model = build_avocado_classifier(pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "epoch_times": [],
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    print(f"Beginning training for up to {num_epochs} epochs (Early stopping patience={patience})...")

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        elapsed = time.time() - start_time

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["epoch_times"].append(round(elapsed, 2))

        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} [{elapsed:.1f}s] - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%"
        )

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(model, epoch, val_loss, val_acc, checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch}.")
                break

    # Save history and curves
    with open(experiments_dir / "deep_learning_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    metadata = {
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_acc": round(best_val_acc, 4),
        "total_epochs_trained": len(history["train_loss"]),
        "device": device,
        "learning_rate": lr,
        "patience": patience,
    }
    with open(experiments_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    plot_training_curves(history, save_path=experiments_dir / "training_curves.png")
    print(f"Training history and curves saved to {experiments_dir}")
    return metadata
