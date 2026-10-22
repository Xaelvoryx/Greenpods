"""
PyTorch Dataset and DataLoader implementation for Avocado Ripening Stage Classification.
Supports deterministic training, validation, and testing partitions with label-preserving augmentations.
"""

from pathlib import Path
from typing import Optional, Callable, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

from src.config import (
    CLASSES,
    CLASS_TO_IDX,
    DL_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    BATCH_SIZE,
    NUM_WORKERS,
)


def get_transforms(split: str = "train") -> transforms.Compose:
    """
    Construct torchvision transform pipeline.
    Training uses moderate, label-preserving spatial and color jitter augmentations.
    Validation and test use strictly deterministic resize and normalization.
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize(DL_IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(DL_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class AvocadoDataset(Dataset):
    """PyTorch Dataset loading images and canonical ripening stage labels from manifest."""

    def __init__(
        self,
        df: pd.DataFrame,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        self.split = split
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform if transform is not None else get_transforms(split)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        img_path = row["file_path"]
        class_idx = int(row["class_idx"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback to black image if corrupted
            image = Image.new("RGB", DL_IMAGE_SIZE, color=(0, 0, 0))

        if self.transform:
            tensor = self.transform(image)
        else:
            tensor = transforms.ToTensor()(image)

        return tensor, class_idx, str(img_path)


def create_dataloaders(
    manifest_df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create Train, Validation, and Test PyTorch DataLoaders."""
    train_dataset = AvocadoDataset(manifest_df, split="train")
    val_dataset = AvocadoDataset(manifest_df, split="val")
    test_dataset = AvocadoDataset(manifest_df, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
