"""
Unit tests for Deep Learning MobileNetV2 architecture, output shape, and forward pass.
"""

import pytest
import torch
from src.config import CLASSES
from src.dl_model import build_avocado_classifier, save_checkpoint, load_dl_model


def test_mobilenetv2_architecture():
    """Verify 4-class output dimension and forward pass."""
    model = build_avocado_classifier(num_classes=4, pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)

    assert output.shape == (2, 4)
    assert not torch.isnan(output).any()


def test_mobilenetv2_checkpoint_save_load(tmp_path):
    """Verify saving and reloading weights."""
    model = build_avocado_classifier(num_classes=4, pretrained=False)
    ckpt_path = tmp_path / "test_mobilenetv2.pth"

    save_checkpoint(model, epoch=1, val_loss=0.5, val_acc=0.85, save_path=ckpt_path)
    assert ckpt_path.exists()

    loaded_model = load_dl_model(ckpt_path, device="cpu")
    dummy_input = torch.randn(1, 3, 224, 224)
    out = loaded_model(dummy_input)
    assert out.shape == (1, 4)
