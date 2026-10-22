"""
Unit tests for Classical ML feature extraction, training, and artifact persistence.
"""

from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from src.classical_model import (
    extract_classical_features,
    build_classical_pipeline,
    save_classical_model,
    load_classical_model,
)


def test_classical_feature_dimensions(tmp_path):
    """Test feature extractor generates 74-dimensional finite vector."""
    img_path = tmp_path / "test_avocado.jpg"
    img = Image.new("RGB", (128, 128), color=(34, 139, 34))
    img.save(img_path)

    feats = extract_classical_features(str(img_path))
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (74,)
    assert np.all(np.isfinite(feats))
    assert not np.any(np.isnan(feats))


def test_classical_pipeline_fit_predict(tmp_path):
    """Test fitting and serialization on synthetic data."""
    X = np.random.randn(20, 74)
    y = np.random.randint(0, 4, size=20)

    pipeline = build_classical_pipeline(c_param=1.0)
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == 20

    # Save and reload test
    model_file = tmp_path / "test_model.joblib"
    save_classical_model(pipeline, save_path=model_file)
    assert model_file.exists()

    loaded = load_classical_model(model_file)
    assert "pipeline" in loaded
    loaded_preds = loaded["pipeline"].predict(X)
    np.testing.assert_array_equal(preds, loaded_preds)
