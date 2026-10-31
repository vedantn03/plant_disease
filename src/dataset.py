"""
dataset.py
==========
PyTorch Dataset and DataLoader factories for PlantVillage.

Key features:
  - Loads from split CSV manifests (train/val/test).
  - Applies configurable augmentation to train; only resize+normalize for val/test.
  - Returns (image_tensor, class_idx) pairs.
  - Computes per-class weights for weighted loss.
  - Never exposes filenames, source codes, or crop metadata to the model.
"""

import csv
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ImageNet mean/std — standard for pretrained torchvision models
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(split: Literal["train", "val", "test"], cfg) -> transforms.Compose:
    """
    Return transforms for the given split.
    Train: augmentation + normalize.
    Val/Test: resize + center-crop + normalize only.
    """
    img_size: int = cfg.data.image_size
    aug = cfg.augmentation

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=aug.hflip_prob),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=aug.rotation_degrees),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
            transforms.ColorJitter(
                brightness=aug.color_jitter.brightness,
                contrast=aug.color_jitter.contrast,
                saturation=aug.color_jitter.saturation,
                hue=aug.color_jitter.hue,
            ),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),  # e.g. 256 for img_size=224
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])


class PlantVillageDataset(Dataset):
    """
    Loads images from a split manifest CSV.

    The model NEVER sees filenames, source codes, or crop metadata —
    only the pixel tensor and the class index.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        transform: transforms.Compose,
        subsample_per_class: int | None = None,
        seed: int = 42,
        class_to_idx: dict[str, int] | None = None,
    ) -> None:
        """
        Args:
            manifest_path: Path to split CSV (from audit_and_split.py).
            transform: torchvision transform to apply.
            subsample_per_class: If set, cap each class to this many images
                                  (stratified sub-sample for fast training runs).
            seed: RNG seed for reproducible sub-sampling.
            class_to_idx: Optional pre-built mapping (pass from train set to val/test
                           so indices are consistent).
        """
        self.transform = transform
        self.rng = np.random.default_rng(seed)

        # Read manifest
        records = _read_csv(manifest_path)

        # Build / use class index mapping (alphabetical for consistency)
        all_classes = sorted({r["class_name"] for r in records})
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            self.class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Optional sub-sampling per class
        if subsample_per_class is not None:
            records = self._subsample(records, subsample_per_class)

        self.samples: list[tuple[str, int]] = [
            (r["path"], self.class_to_idx[r["class_name"]]) for r in records
        ]

    def _subsample(self, records: list[dict], cap: int) -> list[dict]:
        class_buckets: dict[str, list[dict]] = {}
        for r in records:
            class_buckets.setdefault(r["class_name"], []).append(r)

        result = []
        for cls, recs in class_buckets.items():
            if len(recs) <= cap:
                result.extend(recs)
            else:
                indices = self.rng.choice(len(recs), size=cap, replace=False)
                result.extend([recs[i] for i in indices])
        return result

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

    def class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for weighted cross-entropy loss.

        Returns:
            1D float tensor of shape (num_classes,).
        """
        counts = np.zeros(len(self.class_to_idx), dtype=np.float64)
        for _, label in self.samples:
            counts[label] += 1
        # Replace zeros with 1 to avoid division by zero
        counts = np.where(counts == 0, 1, counts)
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(self.class_to_idx)
        return torch.tensor(weights, dtype=torch.float32)


def _read_csv(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_dataloaders(
    splits_dir: str | Path,
    cfg,
    subsample_per_class: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    """
    Create train, val, test DataLoaders from split manifests.

    Returns:
        (train_loader, val_loader, test_loader, class_to_idx)
    """
    splits_dir = Path(splits_dir)
    seed = cfg.seed

    train_ds = PlantVillageDataset(
        splits_dir / "train.csv",
        transform=get_transforms("train", cfg),
        subsample_per_class=subsample_per_class,
        seed=seed,
    )
    val_ds = PlantVillageDataset(
        splits_dir / "val.csv",
        transform=get_transforms("val", cfg),
        subsample_per_class=None,  # always use full val
        seed=seed,
        class_to_idx=train_ds.class_to_idx,
    )
    test_ds = PlantVillageDataset(
        splits_dir / "test.csv",
        transform=get_transforms("test", cfg),
        subsample_per_class=None,  # always use full test
        seed=seed,
        class_to_idx=train_ds.class_to_idx,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        persistent_workers=cfg.training.num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        persistent_workers=cfg.training.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        persistent_workers=cfg.training.num_workers > 0,
    )

    return train_loader, val_loader, test_loader, train_ds.class_to_idx
