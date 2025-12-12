"""
DataModule classes for YOLO using PyTorch Lightning.
"""

import logging
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from lit_yolo.data.datasets import DetDataset, OBBDataset
from lit_yolo.data.utils import (
    SYNTHETIC_COLORS,
    SYNTHETIC_SHAPES,
    annotate_batch_images,
    determine_num_classes,
    generate_synthetic_sample,
    read_class_names_from_yaml,
)

logger = logging.getLogger(__name__)


def show_images_in_grid(
    images: list[np.ndarray],
    output_path: str | Path | None = None,
) -> np.ndarray:
    """Display or save images in a grid using matplotlib subplots.

    Args:
        images: List of images in BGR format (H, W, 3).
        output_path: Optional path to save the grid. If None, only returns array.

    Returns:
        Grid image as numpy array in BGR format.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required for visualization. Install it with: pip install matplotlib") from e

    # Determine grid layout
    n = len(images)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Plot each image
    for idx, (ax, img) in enumerate(zip(axes, images)):
        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.axis("off")

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info(f"Saved visualization to {output_path}")

    # Convert figure to numpy array for return value
    fig.canvas.draw()
    grid = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    grid = grid.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # Convert RGB to BGR for consistency
    grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)

    # Close figure to free memory
    plt.close(fig)

    return grid


class BaseDataModule(LightningDataModule):
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
        output_path: str | Path | None = None,
        batch_idx: int = 0,
    ) -> np.ndarray:
        """Visualize a batch from the dataset with annotations.

        Creates a grid image showing all samples in the specified batch with drawn bounding boxes
        and class labels. Class names are automatically loaded from dataset YAML file if available,
        otherwise class indices are used.

        Args:
            split: Dataset split to visualize ("train" or "val").
            output_path: Optional path to save the visualization. If None, only returns the image.
            batch_idx: Index of the batch to visualize (default: 0 for first batch).

        Returns:
            Grid image as numpy array in BGR format.
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
            raise ValueError(f"Batch index {batch_idx} out of range")

        # Annotate images in the batch
        annotated_images = annotate_batch_images(batch, self._draw_boxes_on_image, self.class_names)

        # Create and return grid visualization
        return show_images_in_grid(annotated_images, output_path)

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


class OBBDataModule(BaseDataModule):
    """Lightning DataModule for OBB datasets - handles all data setup."""

    def setup(self, stage: str | None = None):
        """Setup OBB datasets for training and validation."""
        nc = self.num_classes  # Triggers detection if needed
        self.train_ds = OBBDataset(self.data_root, "train", self.img_size, nc)
        self.val_ds = OBBDataset(self.data_root, "val", self.img_size, nc)

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
        from lit_yolo.data.utils import draw_obb_on_image

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
        from lit_yolo.data.utils import draw_bboxes_on_image

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
        seed=seed,
    )

    logger.info(f"Synthetic dataset created successfully at: {dataset_path}")


def show_dataset(
    data: str,
    output: str | None = None,
    split: Literal["train", "val"] = "train",
    batch_size: int = 8,
    batch_idx: int = 0,
    img_size: int = 640,
    num_workers: int = 4,
    num_classes: int | None = None,
) -> None:
    """Visualize a batch from the dataset with annotations.

    Creates a grid image showing all samples in the specified batch with drawn
    bounding boxes (oriented or axis-aligned) and class labels. Class names are
    automatically loaded from the dataset YAML file if available, otherwise class
    indices are used. If output path is not provided, displays the image in a matplotlib window.

    Args:
        data: Path to dataset root directory (with images/ and labels/ subdirs).
        output: Output path for visualization image. If None, shows in matplotlib window.
        split: Dataset split to visualize ("train" or "val").
        batch_size: Number of images in the batch to visualize.
        batch_idx: Index of batch to visualize (0 for first batch).
        img_size: Image size for loading.
        num_workers: Number of dataloader workers.
        num_classes: Number of classes (auto-detected if None).

    Examples:
        From command line::

            lit-yolo show dataset --data /path/to/dataset --output viz.jpg
            lit-yolo show dataset --data /path/to/dataset --split val --batch_size 4
            lit-yolo show dataset --data /path/to/dataset
    """
    # Use OBB datamodule as it supports both oriented and plain bounding boxes
    logger.info(f"Loading dataset from {data}...")
    datamodule = OBBDataModule(
        data=data,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        num_classes=num_classes,
    )

    # Setup datamodule
    datamodule.setup("fit")
    logger.info(f"Detected {datamodule.num_classes} classes")

    # Log class names if available
    if datamodule.class_names:
        logger.info(f"Loaded class names from dataset YAML: {datamodule.class_names}")

    # Visualize batch
    logger.info(f"Visualizing {split} batch {batch_idx}...")
    grid = datamodule.visualize_batch(
        split=split,
        output_path=output,
        batch_idx=batch_idx,
    )

    if output is not None:
        logger.info(f"✓ Visualization saved to {output}")
    else:
        # Display in matplotlib window
        try:
            import matplotlib
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib is required to display images. Install it with: pip install matplotlib")
            logger.info("Or specify --output to save to a file instead.")
            return

        # Check if we have a GUI backend available
        backend = matplotlib.get_backend()
        if backend.lower() in ('agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template'):
            logger.warning(
                f"No GUI backend available (current: {backend}). Cannot display interactive window."
            )
            logger.info("Running in headless mode. Please specify --output to save to a file instead.")
            return

        # Convert BGR to RGB for matplotlib
        grid_rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 8))
        plt.imshow(grid_rgb)
        plt.axis("off")
        plt.title(f"Dataset: {split} batch {batch_idx}")
        plt.tight_layout()
        logger.info("Displaying visualization in matplotlib window...")
        try:
            plt.show()
        except Exception as e:
            logger.error(f"Failed to display window: {e}")
            logger.info("Please specify --output to save to a file instead.")

    logger.info(f"  Grid shape: {grid.shape}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info("Visualization complete!")
