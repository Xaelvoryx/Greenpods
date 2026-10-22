"""
Script 5: Evaluate Models (scripts/evaluate_models.py)
Loads the held-out test split, runs both Classical ML (SVM) and Deep Learning (MobileNetV2),
computes comprehensive comparative metrics (Accuracy, Macro Precision/Recall/F1, Latency, Size),
and outputs comparison tables, confusion matrices, and misclassification error audits.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    SPLIT_MANIFEST_PATH,
    CLASSICAL_MODEL_PATH,
    DL_MODEL_PATH,
    EVALUATION_OUTPUTS_DIR,
    DEVICE,
)
from src.preprocessing import load_split_manifest
from src.evaluation import (
    evaluate_classical,
    evaluate_deep_learning,
    generate_model_comparison,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Classical ML and Deep Learning models on held-out test set.")
    parser.add_argument("--manifest", type=str, default=str(SPLIT_MANIFEST_PATH), help="Path to split manifest JSON")
    parser.add_argument("--classical-model", type=str, default=str(CLASSICAL_MODEL_PATH), help="Path to classical model artifact")
    parser.add_argument("--dl-model", type=str, default=str(DL_MODEL_PATH), help="Path to MobileNetV2 checkpoint")
    parser.add_argument("--output-dir", type=str, default=str(EVALUATION_OUTPUTS_DIR), help="Output directory for reports")
    parser.add_argument("--device", type=str, default=DEVICE, help="Evaluation device")
    args = parser.parse_args()

    print(f"Loading test split from {args.manifest}...")
    manifest_df = load_split_manifest(Path(args.manifest))
    test_df = manifest_df[manifest_df["split"] == "test"].reset_index(drop=True)
    print(f"Found {len(test_df)} held-out test samples.")

    out_dir = Path(args.output_dir)

    # 1. Evaluate Classical Model
    print("\n" + "=" * 50)
    print(" EVALUATING APPROACH 1: CLASSICAL CV + SVM")
    print("=" * 50)
    c_report, c_cm, c_latency = evaluate_classical(test_df, model_path=Path(args.classical_model), output_dir=out_dir)
    print(f"Classical Test Accuracy: {c_report['accuracy']*100:.2f}%")
    print(f"Classical Macro F1:      {c_report['macro avg']['f1-score']:.4f}")
    print(f"Classical Latency:       {c_latency:.2f} ms")

    # 2. Evaluate Deep Learning Model
    print("\n" + "=" * 50)
    print(" EVALUATING APPROACH 2: MOBILENETV2 TRANSFER LEARNING")
    print("=" * 50)
    dl_report, dl_cm, dl_latency = evaluate_deep_learning(test_df, checkpoint_path=Path(args.dl_model), output_dir=out_dir, device=args.device)
    print(f"MobileNetV2 Test Accuracy: {dl_report['accuracy']*100:.2f}%")
    print(f"MobileNetV2 Macro F1:      {dl_report['macro avg']['f1-score']:.4f}")
    print(f"MobileNetV2 Latency:       {dl_latency:.2f} ms")

    # 3. Generate Comparative Summary Table
    print("\n" + "=" * 50)
    print(" GENERATING COMPARATIVE BENCHMARK")
    print("=" * 50)
    comparison_df = generate_model_comparison(
        c_report,
        dl_report,
        c_latency,
        dl_latency,
        output_dir=out_dir,
    )
    print("\nModel Comparison Table:")
    print(comparison_df.to_string(index=False))
    print(f"\nAll evaluation artifacts saved in: {out_dir}")


if __name__ == "__main__":
    main()
