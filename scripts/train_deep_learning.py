"""
Script 4: Train Deep Learning (scripts/train_deep_learning.py)
Configures PyTorch MobileNetV2 transfer learning, applies training augmentations,
runs training and validation loops with early stopping, records training curves,
and saves the best model checkpoint.
"""

import argparse
import sys
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    SPLIT_MANIFEST_PATH,
    DL_MODEL_PATH,
    EXPERIMENTS_OUTPUTS_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    DEVICE,
    RANDOM_SEED,
)
from src.preprocessing import load_split_manifest
from src.dataset import create_dataloaders
from src.training import train_model
from src.reproducibility import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 Transfer Learning Model on Avocado Ripening Stages.")
    parser.add_argument("--manifest", type=str, default=str(SPLIT_MANIFEST_PATH), help="Path to split manifest JSON")
    parser.add_argument("--output-model", type=str, default=str(DL_MODEL_PATH), help="Path to save best PyTorch checkpoint")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Adam learning rate")
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE, help="Early stopping patience")
    parser.add_argument("--device", type=str, default=DEVICE, help="Execution device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 quick epoch for sanity check")
    args = parser.parse_args()

    seed_everything(args.seed)

    print(f"Loading split manifest from {args.manifest}...")
    manifest_df = load_split_manifest(Path(args.manifest))

    train_loader, val_loader, _ = create_dataloaders(
        manifest_df,
        batch_size=args.batch_size,
    )
    print(f"Created DataLoaders: Train ({len(train_loader.dataset)} items), Val ({len(val_loader.dataset)} items)")

    epochs = 1 if args.smoke_test else args.epochs
    patience = 1 if args.smoke_test else args.patience

    print(f"Starting training on {args.device} for {epochs} epochs...")
    metadata = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        lr=args.lr,
        patience=patience,
        device=args.device,
        checkpoint_path=Path(args.output_model),
        experiments_dir=EXPERIMENTS_OUTPUTS_DIR,
    )

    print("\nTraining Completed Successfully:")
    print(f"  Best Epoch:     {metadata['best_epoch']}")
    print(f"  Best Val Loss:  {metadata['best_val_loss']}")
    print(f"  Best Val Acc:   {metadata['best_val_acc']*100:.2f}%")
    print(f"  Checkpoint:     {args.output_model}")


if __name__ == "__main__":
    main()
