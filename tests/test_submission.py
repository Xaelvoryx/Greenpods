"""
Integration smoke test verifying the end-to-end pipeline:
Synthetic dataset -> discovery -> splitting -> classical feature extraction ->
classical training -> MobileNetV2 forward pass -> evaluation -> model saving -> inference.
"""

from pathlib import Path
import pytest
import numpy as np
import torch
from PIL import Image

from src.config import CLASSES
from src.preprocessing import create_stratified_manifest, load_split_manifest
from src.classical_model import (
    extract_classical_features,
    build_classical_pipeline,
    save_classical_model,
)
from src.dl_model import build_avocado_classifier, save_checkpoint
from src.evaluation import evaluate_classical, generate_model_comparison
from src.inference import predict_single_image


def test_full_pipeline_synthetic_smoke(tmp_path):
    """Run full pipeline smoke test on synthetic fixture."""
    data_dir = tmp_path / "data"
    for c in CLASSES:
        cdir = data_dir / c
        cdir.mkdir(parents=True)
        for i in range(10):
            img = Image.new("RGB", (64, 64), color=(i * 15, 80, 40))
            img.save(cdir / f"sample_{i}.png")

    # 1. Stratified manifest
    manifest_path = tmp_path / "manifest.json"
    create_stratified_manifest([data_dir], output_path=manifest_path, seed=42)
    manifest_df = load_split_manifest(manifest_path)
    train_df = manifest_df[manifest_df["split"] == "train"]
    test_df = manifest_df[manifest_df["split"] == "test"]

    # 2. Classical ML
    X_train = np.array([extract_classical_features(p) for p in train_df["file_path"]])
    y_train = np.array(train_df["class_idx"])
    pipeline = build_classical_pipeline(c_param=1.0)
    pipeline.fit(X_train, y_train)

    c_model_path = tmp_path / "c_model.joblib"
    save_classical_model(pipeline, save_path=c_model_path)

    # 3. DL Model Mock Checkpoint
    dl_model = build_avocado_classifier(num_classes=4, pretrained=False)
    dl_model_path = tmp_path / "dl_model.pth"
    save_checkpoint(dl_model, epoch=1, val_loss=0.8, val_acc=0.75, save_path=dl_model_path)

    # 4. Evaluation
    c_report, c_cm, c_lat = evaluate_classical(test_df, model_path=c_model_path, output_dir=tmp_path)
    assert "accuracy" in c_report

    # 5. Inference
    test_sample = test_df.iloc[0]["file_path"]
    demo_out = predict_single_image(test_sample, classical_model_path=c_model_path, dl_model_path=dl_model_path)
    assert demo_out["classical"]["status"] == "SUCCESS"
    assert demo_out["deep_learning"]["status"] == "SUCCESS"
