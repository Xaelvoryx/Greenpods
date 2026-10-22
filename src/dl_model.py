"""
Approach 2: Deep Learning Transfer Learning Architecture (MobileNetV2).
Builds a lightweight 4-class classifier with pretrained ImageNet backbone
and custom classification head for avocado ripening stage detection.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torchvision import models

from src.config import CLASSES, DL_MODEL_PATH


def build_avocado_classifier(
    num_classes: int = len(CLASSES),
    pretrained: bool = True,
    dropout_rate: float = 0.2,
) -> nn.Module:
    """
    Construct MobileNetV2 architecture with pretrained ImageNet weights.
    Freezes initial feature backbone and attaches custom 4-class linear head.
    """
    if pretrained:
        try:
            weights = models.MobileNet_V2_Weights.DEFAULT
            model = models.mobilenet_v2(weights=weights)
        except Exception as e:
            print(f"Warning: Could not fetch online pretrained weights ({e}). Initializing backbone.")
            model = models.mobilenet_v2(weights=None)
    else:
        model = models.mobilenet_v2(weights=None)

    # Freeze backbone feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier head: in_features -> Linear(128) -> ReLU -> Dropout -> Linear(num_classes)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=dropout_rate),
        nn.Linear(128, num_classes)
    )
    return model


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    val_loss: float,
    val_acc: float,
    save_path: Path = DL_MODEL_PATH,
) -> None:
    """Persist deep learning model checkpoint with metadata."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "classes": CLASSES,
        "state_dict": model.state_dict(),
        "arch": "MobileNetV2",
    }
    torch.save(state, save_path)
    print(f"Saved best checkpoint to {save_path} (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f})")


def load_dl_model(
    checkpoint_path: Path = DL_MODEL_PATH,
    device: str = "cpu",
) -> nn.Module:
    """Load saved MobileNetV2 checkpoint into evaluation mode."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Deep learning checkpoint not found at {checkpoint_path}")

    model = build_avocado_classifier(num_classes=len(CLASSES), pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model
