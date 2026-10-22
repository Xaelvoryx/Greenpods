"""
Unit tests for inference module, error handling, missing artifacts, and corrupted inputs.
"""

from pathlib import Path
import pytest
from PIL import Image

from src.inference import predict_single_image


def test_inference_missing_image():
    """Verify error raised on non-existent input image."""
    with pytest.raises(FileNotFoundError):
        predict_single_image("non_existent_file.jpg")


def test_inference_corrupted_image(tmp_path):
    """Verify error raised on corrupt image."""
    bad_img = tmp_path / "broken.jpg"
    with open(bad_img, "wb") as f:
        f.write(b"not an image")

    with pytest.raises(ValueError):
        predict_single_image(str(bad_img))


def test_inference_graceful_missing_models(tmp_path):
    """Verify graceful status reporting when models are unbuilt."""
    valid_img = tmp_path / "valid.jpg"
    Image.new("RGB", (100, 100), color=(10, 50, 20)).save(valid_img)

    res = predict_single_image(
        str(valid_img),
        classical_model_path=tmp_path / "missing_c.joblib",
        dl_model_path=tmp_path / "missing_dl.pth",
    )

    assert "ARTIFACT_MISSING" in res["classical"]["status"]
    assert "ARTIFACT_MISSING" in res["deep_learning"]["status"]
