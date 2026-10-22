"""
Unit tests for evaluation metrics computation, confusion matrix formatting, and comparison table.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.config import CLASSES
from src.evaluation import generate_model_comparison


def test_model_comparison_table_structure(tmp_path):
    """Verify comparison table generation and column structure."""
    c_report = {
        "accuracy": 0.88,
        "macro avg": {"precision": 0.87, "recall": 0.86, "f1-score": 0.865},
    }
    dl_report = {
        "accuracy": 0.94,
        "macro avg": {"precision": 0.93, "recall": 0.94, "f1-score": 0.935},
    }

    df = generate_model_comparison(
        c_report,
        dl_report,
        classical_latency_ms=12.5,
        dl_latency_ms=38.2,
        output_dir=tmp_path,
    )

    assert "Metric" in df.columns
    assert "Classical ML" in df.columns
    assert "MobileNetV2" in df.columns
    assert len(df) == 6
    assert (tmp_path / "model_comparison.csv").exists()
    assert (tmp_path / "model_comparison.json").exists()
