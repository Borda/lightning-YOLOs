"""
DataModule classes for YOLO using PyTorch Lightning.
"""

import logging
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from lit_yolo.data.datasets import YOLODetDataset, YOLOOBBDataset
from lit_yolo.data.utils import (
    SYNTHETIC_COLORS,
    SYNTHETIC_SHAPES,
    determine_num_classes,
    generate_synthetic_sample,
)

logger = logging.getLogger(__name__)


# DATA MODULES
# =============================================================================


class BaseYOLODataModule(LightningDataModule):
    """
    Base Lightning DataModule for YOLO datasets.

    This abstract base class provides common functionality for YOLO dataset DataModules,
    including dataloader configuration, automatic class detection, and standardized
    interface for training and validation data loading.

    Subclasses must implement the following methods:
        - setup(stage): Prepare and assign self.train_ds and self.val_ds datasets for the given stage.
        - _collate(batch): Collate function for batching samples, used by the dataloaders.

    Common functionality provided:
        - Dataloader creation for training and validation with consistent configuration.
        - Automatic detection of the number of classes from the dataset if not provided.
        - Standardized interface for extending to new YOLO variants (e.g., OBB, detection).

    To extend for a new YOLO variant, subclass this class and implement the required methods.
    """

    train_ds: Dataset | None = None
    val_ds: Dataset | None = None
    test_ds: Dataset | None = None

    def __init__(
        self, data: str, img_size: int = 640, batch_size: int = 8, num_workers: int = 4, num_classes: int | None = None
    ):
        """Initialize base YOLO data module.

        Args:
            data: Path to dataset root directory.
            img_size: Target image size.
            batch_size: Batch size for dataloaders.
            num_workers: Number of workers for dataloaders.
            num_classes: Number of classes (auto-detected if None).
        """
        super().__init__()
        self.data_root = Path(data)
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._num_classes = num_classes

    @property
    def num_classes(self) -> int:
        """Get number of classes, auto-detecting if not provided."""
        if self._num_classes is None:
            self._num_classes = determine_num_classes(self.data_root)
        return self._num_classes

    def setup(self, stage: str | None = None):
        """Setup datasets. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement setup")

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self._collate,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate,
            persistent_workers=self.num_workers > 0,
        )

    def _collate(self, batch: list[tuple]) -> dict[str, Any]:
        """Collate function for batching. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _collate")

    @staticmethod
    def create_synthetic_dataset(
        root: Path | str,
        num_samples: int = 100,
        split_ratio: float = 0.8,
        img_size: int = 640,
        class_mode: Literal["shape", "color"] = "shape",
        min_objects: int = 3,
        max_objects: int = 5,
        min_size_ratio: float = 0.1,
        max_size_ratio: float = 0.2,
        seed: int = 42,
    ) -> Path:
        """Create a synthetic dataset with basic geometric shapes for testing.

        Generates images containing basic shapes (square, triangle, circle) with different colors
        (red, green, blue). Classes can be defined by shape or color depending on class_mode.

        Args:
            root: Root directory where dataset will be created.
            num_samples: Total number of samples to generate.
            split_ratio: Ratio of training samples (e.g., 0.8 means 80% train, 20% val).
            img_size: Size of generated images (square).
            class_mode: Classification mode - "shape" or "color".
            min_objects: Minimum number of objects per image.
            max_objects: Maximum number of objects per image.
            min_size_ratio: Minimum object size as ratio of image size (default 0.1 = 10%).
            max_size_ratio: Maximum object size as ratio of image size (default 0.2 = 20%).
            seed: Random seed for reproducibility.

        Returns:
            Path to the created dataset root directory.

        Examples:
            >>> import tempfile
            >>> from pathlib import Path
            >>> # Create synthetic dataset
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     root = Path(tmpdir) / "synthetic"
            ...     dataset_path = BaseYOLODataModule.create_synthetic_dataset(
            ...         root, num_samples=10, split_ratio=0.7
            ...     )
            ...     len(list((dataset_path / "images" / "train").glob("*.jpg")))
            7
            ...     len(list((dataset_path / "images" / "val").glob("*.jpg")))
            3
        """
        root = Path(root)
        np.random.seed(seed)

        # Calculate split
        num_train = int(num_samples * split_ratio)
        num_val = num_samples - num_train

        # Create directory structure
        for split in ["train", "val"]:
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)

        # Generate datasets for both splits
        for split, num_imgs in [("train", num_train), ("val", num_val)]:
            logger.info(f"Generating {num_imgs} {split} images...")
            for i in range(num_imgs):
                img_path = root / "images" / split / f"img_{i:05d}.jpg"
                label_path = root / "labels" / split / f"img_{i:05d}.txt"

                # Generate image and labels
                img, labels = generate_synthetic_sample(
                    img_size=img_size,
                    min_objects=min_objects,
                    max_objects=max_objects,
                    class_mode=class_mode,
                    min_size_ratio=min_size_ratio,
                    max_size_ratio=max_size_ratio,
                )

                # Save image
                cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                # Save labels
                with open(label_path, "w") as f:
                    for cls, cx, cy, w, h in labels:
                        f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        class_names = SYNTHETIC_SHAPES if class_mode == "shape" else list(SYNTHETIC_COLORS.keys())
        logger.info(f"Synthetic dataset created at {root}")
        logger.info(f"Class mode: {class_mode} (3 classes: {', '.join(class_names)})")
        logger.info(f"Train: {num_train} images, Val: {num_val} images")

        return root


class OBBDataModule(BaseYOLODataModule):
    """Lightning DataModule for OBB datasets - handles all data setup."""

    def setup(self, stage: str | None = None):
        """Setup OBB datasets for training and validation."""
        nc = self.num_classes  # Triggers detection if needed
        self.train_ds = YOLOOBBDataset(self.data_root, "train", self.img_size, nc)
        self.val_ds = YOLOOBBDataset(self.data_root, "val", self.img_size, nc)

    @staticmethod
    def _collate(batch: list[tuple]) -> dict[str, Any]:
        """Collate function for OBB batches with 5 bbox parameters (cx, cy, w, h, angle)."""
        imgs, batch_idx, cls_list, bbox_list = [], [], [], []

        for i, (img, labels) in enumerate(batch):
            imgs.append(img)
            if labels.numel() > 0:
                n = labels.shape[0]
                batch_idx.append(torch.full((n,), i, dtype=torch.long))
                cls_list.append(labels[:, 0:1])
                bbox_list.append(labels[:, 1:6])

        return {
            "img": torch.stack(imgs),
            "batch_idx": torch.cat(batch_idx) if batch_idx else torch.empty(0, dtype=torch.long),
            "cls": torch.cat(cls_list) if cls_list else torch.empty(0, 1),
            "bboxes": torch.cat(bbox_list) if bbox_list else torch.empty(0, 5),
        }


class DetDataModule(BaseYOLODataModule):
    """Lightning DataModule for standard detection datasets - handles all data setup."""

    def setup(self, stage: str | None = None):
        """Setup standard detection datasets for training and validation."""
        nc = self.num_classes  # Triggers detection if needed
        self.train_ds = YOLODetDataset(self.data_root, "train", self.img_size, nc)
        self.val_ds = YOLODetDataset(self.data_root, "val", self.img_size, nc)

    @staticmethod
    def _collate(batch: list[tuple]) -> dict[str, Any]:
        """Collate function for standard detection batches with 4 bbox parameters (cx, cy, w, h)."""
        imgs, batch_idx, cls_list, bbox_list = [], [], [], []

        for i, (img, labels) in enumerate(batch):
            imgs.append(img)
            if labels.numel() > 0:
                n = labels.shape[0]
                batch_idx.append(torch.full((n,), i, dtype=torch.long))
                cls_list.append(labels[:, 0:1])
                bbox_list.append(labels[:, 1:5])  # Standard detection: 4 bbox params

        return {
            "img": torch.stack(imgs),
            "batch_idx": torch.cat(batch_idx) if batch_idx else torch.empty(0, dtype=torch.long),
            "cls": torch.cat(cls_list) if cls_list else torch.empty(0, 1),
            "bboxes": torch.cat(bbox_list) if bbox_list else torch.empty(0, 4),  # 4 params: cx, cy, w, h
        }


def create_synthetic_dataset(
    output: str = "./synthetic_dataset",
    num_samples: int = 100,
    split_ratio: float = 0.8,
    img_size: int = 640,
    class_mode: str = "shape",
    min_objects: int = 3,
    max_objects: int = 5,
    min_size_ratio: float = 0.1,
    max_size_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    """CLI wrapper for creating a synthetic dataset with geometric shapes.

    Args:
        output: Output directory for the dataset.
        num_samples: Total number of samples to generate.
        split_ratio: Ratio of training samples (e.g., 0.8 = 80% train, 20% val).
        img_size: Size of generated images (square).
        class_mode: Classification mode - "shape" or "color".
        min_objects: Minimum number of objects per image.
        max_objects: Maximum number of objects per image.
        min_size_ratio: Minimum object size as ratio of image size.
        max_size_ratio: Maximum object size as ratio of image size.
        seed: Random seed for reproducibility.
    """
    dataset_path = BaseYOLODataModule.create_synthetic_dataset(
        root=output,
        num_samples=num_samples,
        split_ratio=split_ratio,
        img_size=img_size,
        class_mode=class_mode,
        min_objects=min_objects,
        max_objects=max_objects,
        min_size_ratio=min_size_ratio,
        max_size_ratio=max_size_ratio,
        seed=seed,
    )

    logger.info(f"Synthetic dataset created successfully at: {dataset_path}")
