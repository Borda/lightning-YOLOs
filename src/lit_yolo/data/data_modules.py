"""DataModule classes for YOLO using PyTorch Lightning."""

import logging
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from lit_yolo.data.datasets import DetDataset, OBBDataset
from lit_yolo.data.utils import (
    SYNTHETIC_COLORS,
    SYNTHETIC_SHAPES,
    determine_num_classes,
    generate_synthetic_sample,
    read_class_names_from_yaml,
)
from lit_yolo.data.visual import annotate_batch_images, draw_obb_on_image, show_images_in_grid

logger = logging.getLogger(__name__)


class BaseDataModule(LightningDataModule):
    """Base Lightning DataModule for YOLO datasets.

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
        self._class_names = None

    @property
    def num_classes(self) -> int:
        """Get number of classes, auto-detecting if not provided."""
        if self._num_classes is None:
            self._num_classes = determine_num_classes(self.data_root)
        return self._num_classes

    @property
    def class_names(self) -> list[str] | None:
        """Get class names from dataset YAML file if available."""
        if self._class_names is None:
            self._class_names = read_class_names_from_yaml(self.data_root)
        return self._class_names

    def setup(self, stage: str | None = None):
        """Setup datasets.

        Must be implemented by subclasses.
        """
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
        """Collate function for batching.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _collate")

    def _draw_boxes_on_image(
        self, img: np.ndarray, bboxes: np.ndarray, class_ids: np.ndarray, class_names: list[str] | None = None
    ) -> np.ndarray:
        """Draw bounding boxes on image. Must be implemented by subclasses.

        Args:
            img: Image array in RGB format with values in [0, 255].
            bboxes: Array of bounding boxes (format depends on subclass).
            class_ids: Array of class indices.
            class_names: Optional list of class names for labels.

        Returns:
            Image with drawn boxes in BGR format.
        """
        raise NotImplementedError("Subclasses must implement _draw_boxes_on_image")

    def visualize_batch(
        self,
        split: Literal["train", "val"] = "train",
        batch_idx: int = 0,
    ) -> tuple[Any, np.ndarray]:
        """Visualize a batch from the dataset with annotations.

        Creates a grid image showing all samples in the specified batch with drawn bounding boxes
        and class labels. Class names are automatically loaded from dataset YAML file if available,
        otherwise class indices are used.

        Args:
            split: Dataset split to visualize ("train" or "val").
            batch_idx: Index of the batch to visualize (default: 0 for first batch).

        Returns:
            Tuple of (fig, axes) matplotlib figure and axes.
            Caller is responsible for saving/showing and closing the figure with plt.close(fig).
        """
        # Ensure dataset is setup
        if self.train_ds is None or self.val_ds is None:
            self.setup("fit")

        # Get the appropriate dataloader
        dataloader = self.train_dataloader() if split == "train" else self.val_dataloader()

        # Get the specified batch
        for i, batch in enumerate(dataloader):
            if i == batch_idx:
                break
        else:
            raise IndexError(f"Batch index {batch_idx} out of range")

        # Annotate images in the batch
        annotated_images = annotate_batch_images(batch, self._draw_boxes_on_image, self.class_names)

        # Create and return grid visualization
        return show_images_in_grid(annotated_images)

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
        overlap_threshold: float = 0.3,
        seed: int = 42,
    ) -> Path:
        """Create a synthetic dataset with basic geometric shapes for testing.

        Generates images containing basic shapes (square, triangle, circle) with different colors
        (red, green, blue). Classes can be defined by shape or color depending on class_mode.

        The overlap_threshold parameter controls object placement to prevent objects from hiding
        behind each other or extending outside image boundaries. This uses IoU-based overlap
        detection - lower thresholds create stricter separation but may place fewer objects.

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
            overlap_threshold: Maximum allowed IoU overlap between objects and with boundaries (0-1).
                              Lower values mean less overlap is allowed. Default is 0.3 (balanced).
                              Use 0.0 for no overlap (strictest), 0.1 for minimal overlap,
                              0.5 for lenient placement allowing denser object packing.
            seed: Random seed for reproducibility.

        Returns:
            Path to the created dataset root directory.

        Examples:
            >>> import tempfile
            >>> from pathlib import Path
            >>> # Create synthetic dataset
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     root = Path(tmpdir) / "synthetic"
            ...     dataset_path = BaseDataModule.create_synthetic_dataset(
            ...         root, num_samples=10, split_ratio=0.7
            ...     )
            ...     nb_train = len(list((dataset_path / "images" / "train").glob("*.jpg")))
            ...     nb_val = len(list((dataset_path / "images" / "val").glob("*.jpg")))
            >>> nb_train, nb_val
            (7, 3)
        """
        # Validate class_mode
        if class_mode not in ("shape", "color"):
            raise ValueError(f"class_mode must be 'shape' or 'color', got '{class_mode}'")

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
            for i in tqdm(range(num_imgs), desc=f"Generating {split} images", unit="img"):
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
                    overlap_threshold=overlap_threshold,
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


class OBBDataModule(BaseDataModule):
    """Lightning DataModule for OBB datasets - handles all data setup."""

    def setup(self, stage: str | None = None):
        """Setup OBB datasets for training and validation."""
        nc = self.num_classes  # Triggers detection if needed
        self.train_ds = OBBDataset(self.data_root, "train", self.img_size, nc)
        self.val_ds = OBBDataset(self.data_root, "val", self.img_size, nc)

    @staticmethod
    def _collate(batch: list[tuple]) -> dict[str, Any]:
        """Collate function for OBB batches with 5 bbox parameters (cx, cy, w,
        h, angle)."""
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

    def _draw_boxes_on_image(
        self, img: np.ndarray, bboxes: np.ndarray, class_ids: np.ndarray, class_names: list[str] | None = None
    ) -> np.ndarray:
        """Draw oriented bounding boxes on image.

        Args:
            img: Image array in RGB format with values in [0, 255].
            bboxes: Array of shape (N, 5) with [cx, cy, w, h, angle] in normalized coordinates.
            class_ids: Array of class indices.
            class_names: Optional list of class names for labels.

        Returns:
            Image with drawn oriented boxes in BGR format.
        """
        return draw_obb_on_image(img, bboxes, class_ids, class_names=class_names)


class DetDataModule(BaseDataModule):
    """Lightning DataModule for standard detection datasets - handles all data setup."""

    def setup(self, stage: str | None = None):
        """Setup standard detection datasets for training and validation."""
        nc = self.num_classes  # Triggers detection if needed
        self.train_ds = DetDataset(self.data_root, "train", self.img_size, nc)
        self.val_ds = DetDataset(self.data_root, "val", self.img_size, nc)

    @staticmethod
    def _collate(batch: list[tuple]) -> dict[str, Any]:
        """Collate function for standard detection batches with 4 bbox
        parameters (cx, cy, w, h)."""
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

    def _draw_boxes_on_image(
        self, img: np.ndarray, bboxes: np.ndarray, class_ids: np.ndarray, class_names: list[str] | None = None
    ) -> np.ndarray:
        """Draw axis-aligned bounding boxes on image.

        Args:
            img: Image array in RGB format with values in [0, 255].
            bboxes: Array of shape (N, 4) with [cx, cy, w, h] in normalized coordinates.
            class_ids: Array of class indices.
            class_names: Optional list of class names for labels.

        Returns:
            Image with drawn boxes in BGR format.
        """
        from lit_yolo.data import draw_bboxes_on_image

        return draw_bboxes_on_image(img, bboxes, class_ids, class_names=class_names)


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
    overlap_threshold: float = 0.3,
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
        overlap_threshold: Maximum allowed IoU overlap between objects and with boundaries (0-1).
                          Lower values mean less overlap is allowed. Default is 0.3.
        seed: Random seed for reproducibility.
    """
    # Validate class_mode
    if class_mode not in ("shape", "color"):
        raise ValueError(f"class_mode must be 'shape' or 'color', got '{class_mode}'")

    dataset_path = BaseDataModule.create_synthetic_dataset(
        root=output,
        num_samples=num_samples,
        split_ratio=split_ratio,
        img_size=img_size,
        class_mode=cast(Literal["shape", "color"], class_mode),
        min_objects=min_objects,
        max_objects=max_objects,
        min_size_ratio=min_size_ratio,
        max_size_ratio=max_size_ratio,
        overlap_threshold=overlap_threshold,
        seed=seed,
    )

    logger.info(f"Synthetic dataset created successfully at: {dataset_path}")
