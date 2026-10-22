"""
Script 6: Run All Pipeline (scripts/run_all.py)
Automated orchestrator that executes the entire end-to-end computer vision workflow in order:
1. Environment verification & seed initialization
2. Dataset discovery & audit (EDA)
3. Deterministic stratified splitting (70/15/15)
4. Approach 1: Classical CV (CIELAB + LBP + SVM) training
5. Approach 2: Deep Learning (MobileNetV2 Transfer Learning) training
6. Model evaluation, comparative metrics & confusion matrix generation
7. Single-image live demonstration verification
"""

import argparse
import sys
import time
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_DATA_DIR,
    SPLIT_MANIFEST_PATH,
    CLASSICAL_MODEL_PATH,
    DL_MODEL_PATH,
    EVALUATION_OUTPUTS_DIR,
    EDA_OUTPUTS_DIR,
    DEVICE,
    RANDOM_SEED,
)
from src.reproducibility import seed_everything, save_environment_metadata
from src.data_audit import run_full_dataset_audit, discover_image_files
from src.preprocessing import create_stratified_manifest, load_split_manifest
from src.classical_model import extract_classical_features, build_classical_pipeline, save_classical_model
from src.dataset import create_dataloaders
from src.training import train_model
from src.evaluation import evaluate_classical, evaluate_deep_learning, generate_model_comparison
from src.inference import predict_single_image
from src.visualization import plot_class_distribution, plot_representative_samples
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Execute complete end-to-end Avocado Ripening Classification pipeline.")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs for MobileNetV2 training")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for DL training")
    parser.add_argument("--smoke-test", action="store_true", help="Run in fast smoke test mode (1 epoch)")
    args = parser.parse_args()

    start_total_time = time.time()
    seed_everything(RANDOM_SEED)

    print("=" * 70)
    print(" AVOCADO RIPENING STAGE DETECTION & CLASSIFICATION SYSTEM")
    print(" MASTER END-TO-END PIPELINE ORCHESTRATOR")
    print("=" * 70)
    print(f"Target Device: {DEVICE}")

    # Step 1: Record Environment Metadata
    print("\n[Step 1/7] Recording Environment Metadata...")
    save_environment_metadata(EDA_OUTPUTS_DIR / "environment_metadata.json")

    # Step 2: Audit Dataset & Quality Analysis
    print("\n[Step 2/7] Running Dataset Audit & EDA Quality Checks...")
    data_dirs = [PROJECT_ROOT / "train", PROJECT_ROOT / "test", RAW_DATA_DIR]
    existing_dirs = [d for d in data_dirs if d.exists()]
    audit_summary = run_full_dataset_audit(existing_dirs, output_dir=EDA_OUTPUTS_DIR)
    plot_class_distribution(
        class_dist_csv=EDA_OUTPUTS_DIR / "class_distribution.csv",
        save_path=EDA_OUTPUTS_DIR / "class_distribution.png",
    )

    # Step 3: Stratified Dataset Splitting
    print("\n[Step 3/7] Generating Deterministic 70/15/15 Stratified Split Manifest...")
    manifest = create_stratified_manifest(existing_dirs, output_path=SPLIT_MANIFEST_PATH, seed=RANDOM_SEED)
    manifest_df = load_split_manifest(SPLIT_MANIFEST_PATH)

    # Step 4: Train Classical ML Model
    print("\n[Step 4/7] Training Approach 1: Classical CV (CIELAB + LBP + SVM)...")
    train_df = manifest_df[manifest_df["split"] == "train"].reset_index(drop=True)
    X_train = []
    y_train = []
    for _, row in train_df.iterrows():
        try:
            X_train.append(extract_classical_features(row["file_path"]))
            y_train.append(row["class_idx"])
        except Exception:
            pass
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    pipeline = build_classical_pipeline(c_param=2.0)
    pipeline.fit(X_train, y_train)
    save_classical_model(pipeline, save_path=CLASSICAL_MODEL_PATH)

    # Step 5: Train Deep Learning Model
    print("\n[Step 5/7] Training Approach 2: Deep Learning (MobileNetV2 Transfer Learning)...")
    train_loader, val_loader, _ = create_dataloaders(manifest_df, batch_size=args.batch_size)
    epochs = 1 if args.smoke_test else args.epochs
    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        checkpoint_path=DL_MODEL_PATH,
    )

    # Step 6: Model Evaluation and Comparison
    print("\n[Step 6/7] Evaluating Both Approaches on Held-Out Test Set...")
    test_df = manifest_df[manifest_df["split"] == "test"].reset_index(drop=True)
    c_report, _, c_lat = evaluate_classical(test_df, model_path=CLASSICAL_MODEL_PATH, output_dir=EVALUATION_OUTPUTS_DIR)
    dl_report, _, dl_lat = evaluate_deep_learning(test_df, checkpoint_path=DL_MODEL_PATH, output_dir=EVALUATION_OUTPUTS_DIR)
    comp_df = generate_model_comparison(c_report, dl_report, c_lat, dl_lat, output_dir=EVALUATION_OUTPUTS_DIR)
    print("\n" + comp_df.to_string(index=False))

    # Step 7: Live Single-Image Inference Test
    print("\n[Step 7/7] Executing Live Single-Image Inference Demonstration...")
    sample_test_img = test_df.iloc[0]["file_path"]
    demo_res = predict_single_image(sample_test_img, classical_model_path=CLASSICAL_MODEL_PATH, dl_model_path=DL_MODEL_PATH)
    print(f"Sample Image: {Path(sample_test_img).name}")
    print(f"  Classical ML Pred:    {demo_res['classical']['prediction']} ({demo_res['classical']['confidence']*100:.1f}%)")
    print(f"  Deep Learning Pred:   {demo_res['deep_learning']['prediction']} ({demo_res['deep_learning']['confidence']*100:.1f}%)")

    total_time = time.time() - start_total_time
    print("\n" + "=" * 70)
    print(f" END-TO-END PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.1f} SECONDS!")
    print("=" * 70)


if __name__ == "__main__":
    main()
