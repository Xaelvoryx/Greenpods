"""
Script 3: Train Classical Model (scripts/train_classical.py)
Loads train split from manifest, extracts CIELAB + HSV + LBP feature vectors,
trains an SVM classifier with StandardScaler and class balancing,
and saves the trained artifact to models/classical/avocado_classical_model.joblib.
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    SPLIT_MANIFEST_PATH,
    CLASSICAL_MODEL_PATH,
    RANDOM_SEED,
)
from src.preprocessing import load_split_manifest
from src.classical_model import (
    extract_classical_features,
    build_classical_pipeline,
    save_classical_model,
)
from src.reproducibility import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Train Classical Machine Learning Model (CIELAB + LBP + SVM).")
    parser.add_argument("--manifest", type=str, default=str(SPLIT_MANIFEST_PATH), help="Path to split manifest JSON")
    parser.add_argument("--output-model", type=str, default=str(CLASSICAL_MODEL_PATH), help="Path to save model artifact")
    parser.add_argument("--c-param", type=float, default=2.0, help="SVM regularization parameter C")
    parser.add_argument("--gamma", type=str, default="scale", help="SVM RBF kernel gamma parameter")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    args = parser.parse_args()

    seed_everything(args.seed)

    print(f"Loading dataset manifest from {args.manifest}...")
    df = load_split_manifest(Path(args.manifest))
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    print(f"Found {len(train_df)} training samples.")

    print("Extracting color (CIELAB a*, HSV) and texture (LBP) features...")
    t0 = time.time()
    X_train = []
    y_train = []

    for idx, row in train_df.iterrows():
        try:
            feats = extract_classical_features(row["file_path"])
            X_train.append(feats)
            y_train.append(row["class_idx"])
        except Exception as e:
            print(f"Error on image {row['file_path']}: {e}")

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    feature_time = time.time() - t0
    print(f"Extracted {X_train.shape[1]}-dimensional features from {len(X_train)} samples in {feature_time:.1f}s.")

    print(f"Fitting StandardScaler + RBF SVM pipeline (C={args.c_param}, gamma={args.gamma})...")
    t1 = time.time()
    pipeline = build_classical_pipeline(c_param=args.c_param, gamma=args.gamma)
    pipeline.fit(X_train, y_train)
    fit_time = time.time() - t1
    print(f"Training completed in {fit_time:.2f}s.")

    save_classical_model(pipeline, save_path=Path(args.output_model))
    print("Classical model training pipeline executed successfully.")


if __name__ == "__main__":
    main()
