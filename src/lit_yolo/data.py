"""
Data utilities, dataset, and data module for YOLO-OBB.
"""

import logging
import math
import warnings
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# =============================================================================
# SYNTHETIC DATASET CONSTANTS
# =============================================================================

# Available shapes for synthetic dataset generation
SYNTHETIC_SHAPES = ["square", "triangle", "circle"]

# Available colors for synthetic dataset generation (RGB format)
SYNTHETIC_COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
}


# =============================================================================
# UTILITIES
# =============================================================================


def determine_num_classes(root: Path) -> int:
    """Scan label files to find max class index + 1.

    Args:
        root: Root directory containing labels/train and labels/val subdirectories.

    Returns:
        Number of classes detected (max class index + 1).

    Raises:
        ValueError: If no valid labels found in the directory.
    """
    max_class, files_scanned = -1, 0
    for split in ["train", "val"]:
        label_dir = root / "labels" / split
        if not label_dir.exists():
            continue
        for lf in label_dir.glob("*.txt"):
            try:
                with open(lf) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            max_class = max(max_class, int(parts[0]))
                files_scanned += 1
            except (OSError, ValueError):
                continue

    if max_class < 0:
        raise ValueError(f"No valid labels found in {root}/labels/")
    logger.info(f"Detected {max_class + 1} classes from {files_scanned} label files")
    return max_class + 1


# Scaling factor for corner coordinates to improve numerical stability in minAreaRect
_CORNER_SCALE = 1000.0


def corners_to_xywhr(corners: np.ndarray) -> tuple[float, float, float, float, float]:
    """Convert 4 corner points to (cx, cy, w, h, angle) format.

    Args:
        corners: Array of shape (4, 2) containing the 4 corner points.

    Returns:
        Tuple of (cx, cy, w, h, angle) where:
        - cx, cy: center coordinates
        - w, h: width and height (w >= h by convention)
        - angle: rotation angle in radians [0, pi/2)

    Examples:
        >>> import numpy as np
        >>> # Square box centered at (0.5, 0.5)
        >>> corners = np.array([[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]], dtype=np.float32)
        >>> cx, cy, w, h, angle = corners_to_xywhr(corners)
        >>> abs(cx - 0.5) < 0.01 and abs(cy - 0.5) < 0.01
        True
        >>> abs(w - 0.2) < 0.01 and abs(h - 0.2) < 0.01
        True
    """
    corners_px = (corners * _CORNER_SCALE).astype(np.float32)
    (cx_px, cy_px), (w_px, h_px), angle_deg = cv2.minAreaRect(corners_px)

    cx, cy = cx_px / _CORNER_SCALE, cy_px / _CORNER_SCALE
    w, h = w_px / _CORNER_SCALE, h_px / _CORNER_SCALE

    if w < h:
        w, h = h, w
        angle_deg += 90

    # Clamp angle to [0, pi/2) with a small epsilon to avoid edge cases where angle == pi/2,
    # which can cause issues in downstream calculations (e.g., with trigonometric functions).
    angle_rad = max(0.0, min((angle_deg % 90) * math.pi / 180.0, math.pi / 2 - 1e-6))
    return cx, cy, w, h, angle_rad


def obb_to_xyxy(obb: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Convert OBB (xywhr) to axis-aligned xyxy bounding box.

    Args:
        obb: Tensor of shape (N, 5) with columns [cx, cy, w, h, angle].
        scale: Scaling factor to apply to coordinates.

    Returns:
        Tensor of shape (N, 4) with columns [x1, y1, x2, y2].

    Examples:
        >>> import torch
        >>> # Empty input
        >>> obb = torch.empty((0, 5))
        >>> result = obb_to_xyxy(obb)
        >>> result.shape
        torch.Size([0, 4])
        >>> # Single axis-aligned box
        >>> obb = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.0]])
        >>> result = obb_to_xyxy(obb, scale=1.0)
        >>> result.shape
        torch.Size([1, 4])
    """
    if len(obb) == 0:
        return torch.empty((0, 4), device=obb.device)

    cx, cy, w, h, angle = obb[:, 0] * scale, obb[:, 1] * scale, obb[:, 2] * scale, obb[:, 3] * scale, obb[:, 4]
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    hw, hh = w / 2, h / 2

    dx = torch.stack([hw, hw, -hw, -hw], dim=1)
    dy = torch.stack([hh, -hh, -hh, hh], dim=1)
    corners_x = cx[:, None] + dx * cos_a[:, None] - dy * sin_a[:, None]
    corners_y = cy[:, None] + dx * sin_a[:, None] + dy * cos_a[:, None]

    return torch.stack(
        [corners_x.min(1).values, corners_y.min(1).values, corners_x.max(1).values, corners_y.max(1).values], dim=1
    )


def xywh_to_xyxy(bbox: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Convert standard bbox (cx, cy, w, h) to xyxy format.

    Args:
        bbox: Tensor of shape (N, 4) with [cx, cy, w, h] format
        scale: Scaling factor for coordinates

    Returns:
        Tensor of shape (N, 4) with [x1, y1, x2, y2] format

    Examples:
        >>> import torch
        >>> # Empty input
        >>> bbox = torch.empty((0, 4))
        >>> result = xywh_to_xyxy(bbox)
        >>> result.shape
        torch.Size([0, 4])
        >>> # Single box: center at (0.5, 0.5), size (0.2, 0.1)
        >>> bbox = torch.tensor([[0.5, 0.5, 0.2, 0.1]])
        >>> result = xywh_to_xyxy(bbox, scale=1.0)
        >>> result
        tensor([[0.4000, 0.4500, 0.6000, 0.5500]])
        >>> # With scaling
        >>> bbox = torch.tensor([[0.5, 0.5, 0.2, 0.1]])
        >>> result = xywh_to_xyxy(bbox, scale=640.0)
        >>> result
        tensor([[256., 288., 384., 352.]])
    """
    if len(bbox) == 0:
        return torch.empty((0, 4), device=bbox.device)

    cx, cy, w, h = bbox[:, 0] * scale, bbox[:, 1] * scale, bbox[:, 2] * scale, bbox[:, 3] * scale
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return torch.stack([x1, y1, x2, y2], dim=1)


# =============================================================================
# SYNTHETIC DATASET GENERATION FUNCTIONS
# =============================================================================


def draw_synthetic_shape(img: np.ndarray, shape: str, color: tuple, center: tuple, size: int) -> np.ndarray:
    """Draw a geometric shape on an image.

    Args:
        img: Input image array to draw on.
        shape: Shape to draw ("square", "triangle", or "circle").
        color: RGB color tuple.
        center: Center position (cx, cy).
        size: Size of the shape.

    Returns:
        Image with drawn shape.
    """
    cx, cy = center
    if shape == "square":
        # Draw filled square
        half_size = size // 2
        pt1 = (cx - half_size, cy - half_size)
        pt2 = (cx + half_size, cy + half_size)
        cv2.rectangle(img, pt1, pt2, color, -1)
    elif shape == "triangle":
        # Draw filled triangle (equilateral pointing up)
        height = int(size * 0.866)  # sqrt(3)/2 for equilateral triangle
        pt1 = (cx, cy - 2 * height // 3)
        pt2 = (cx - size // 2, cy + height // 3)
        pt3 = (cx + size // 2, cy + height // 3)
        pts = np.array([pt1, pt2, pt3], np.int32)
        cv2.fillPoly(img, [pts], color)
    elif shape == "circle":
        # Draw filled circle
        cv2.circle(img, (cx, cy), size // 2, color, -1)
    return img


def generate_synthetic_sample(
    img_size: int,
    num_objects: int,
    class_mode: Literal["shape", "color"],
    min_size_ratio: float,
    max_size_ratio: float,
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    """Generate a single synthetic image with labeled objects.

    Args:
        img_size: Size of the image (square).
        num_objects: Number of objects to generate.
        class_mode: Classification mode ("shape" or "color").
        min_size_ratio: Minimum object size as ratio of image size.
        max_size_ratio: Maximum object size as ratio of image size.

    Returns:
        Tuple of (image, labels) where labels is a list of (class, cx, cy, w, h) tuples.
    """
    # Create blank image with gray background
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 128

    labels = []
    color_names = list(SYNTHETIC_COLORS.keys())

    # Generate objects (cycling through shapes and colors)
    for i in range(num_objects):
        shape = SYNTHETIC_SHAPES[i % len(SYNTHETIC_SHAPES)]
        color_name = color_names[i % len(color_names)]
        color = SYNTHETIC_COLORS[color_name]

        # Determine class based on mode
        if class_mode == "shape":
            cls = SYNTHETIC_SHAPES.index(shape)
        else:  # class_mode == "color"
            cls = color_names.index(color_name)

        # Random position and size - ensure valid ranges even for small image sizes
        min_size = max(1, int(img_size * min_size_ratio))
        max_size = max(min_size + 1, int(img_size * max_size_ratio))
        obj_size = np.random.randint(min_size, max_size)
        margin = obj_size
        cx = np.random.randint(margin, img_size - margin)
        cy = np.random.randint(margin, img_size - margin)

        # Draw shape
        img = draw_synthetic_shape(img, shape, color, (cx, cy), obj_size)

        # Create bounding box (normalized YOLO format: cx, cy, w, h)
        # Add padding to the bounding box to ensure it contains the entire shape
        bbox_padding_factor = 1.2
        box_w = (obj_size * bbox_padding_factor) / img_size
        box_h = (obj_size * bbox_padding_factor) / img_size
        box_cx = cx / img_size
        box_cy = cy / img_size

        # Ensure box is within image bounds
        # Clamp width and height to not exceed image boundaries
        box_w = min(box_w, 1.0)
        box_h = min(box_h, 1.0)
        # Clamp center coordinates to keep box within image
        box_cx = max(box_w / 2, min(1.0 - box_w / 2, box_cx))
        box_cy = max(box_h / 2, min(1.0 - box_h / 2, box_cy))

        labels.append((cls, box_cx, box_cy, box_w, box_h))

    return img, labels


# =============================================================================
# DATASET
# =============================================================================


class BaseYOLODataset(Dataset):
    """
    Abstract base class for YOLO-style datasets, providing common functionality for image loading,
    preprocessing, and letterbox resizing.

    This class is intended to be subclassed for specific YOLO dataset variants (e.g., oriented bounding box,
    custom label formats). It handles:
        - Discovering and loading images from a directory structure (expects 'images/' and 'labels/' subdirs).
        - Preprocessing images (resizing, normalization, letterbox padding).
        - Providing a standard __getitem__ interface returning (image, label) pairs as tensors.

    Subclasses must implement:
        - _load_labels(self, idx: int) -> torch.Tensor
            Loads and returns the label(s) for the image at the given index, in the expected format.

    Example:
        class MyYOLODataset(BaseYOLODataset):
            def _load_labels(self, idx: int) -> torch.Tensor:
                # Custom label loading logic here
                ...
    """

    FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, root: Path, split: str, img_size: int, num_classes: int):
        if not isinstance(img_size, int) or img_size <= 0:
            raise ValueError(f"img_size must be a positive integer, got {img_size}")
        self.img_size, self.num_classes = img_size, num_classes
        self.img_dir = root / "images" / split
        self.label_dir = root / "labels" / split
        self._standard_format_warned = False  # Track if warning has been logged

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        self.img_paths = sorted(p for p in self.img_dir.iterdir() if p.suffix.lower() in self.FORMATS)
        if not self.img_paths:
            raise ValueError(f"No images found in {self.img_dir}")
        logger.info(f"[{split}] Loaded {len(self.img_paths)} images")

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single item from the dataset.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            Tuple of (image, labels) where image is a tensor of shape (3, img_size, img_size)
            with values in [0, 1] and labels format depends on the subclass.

        Raises:
            IndexError: If image fails to load.
        """
        img_path = self.img_paths[idx]

        img = cv2.imread(str(img_path))
        if img is None:
            raise IndexError(f"Failed to load: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        labels = self._load_labels(self.label_dir / f"{img_path.stem}.txt")
        img, scale, (pad_w, pad_h) = self._letterbox(img)

        if labels.numel() > 0:
            new_w, new_h = orig_w * scale, orig_h * scale
            labels[:, 1] = (labels[:, 1] * new_w + pad_w) / self.img_size
            labels[:, 2] = (labels[:, 2] * new_h + pad_h) / self.img_size
            labels[:, 3] *= new_w / self.img_size
            labels[:, 4] *= new_h / self.img_size

        # Convert to tensor: HWC -> CHW format and normalize to [0, 1]
        return torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0), labels

    def _load_labels(self, path: Path) -> torch.Tensor:
        """Load labels from file. Must be implemented by subclasses.

        Args:
            path: Path to the label file.

        Returns:
            Tensor of labels in format specific to the subclass.
        """
        raise NotImplementedError("Subclasses must implement _load_labels")

    def _letterbox(self, img: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Apply letterbox resizing to maintain aspect ratio.

        Args:
            img: Input image as numpy array.

        Returns:
            Tuple of (padded_image, scale, (pad_w, pad_h))
        """
        h, w = img.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w, pad_h = (self.img_size - new_w) // 2, (self.img_size - new_h) // 2
        img_padded = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        img_padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = img_resized
        return img_padded, scale, (pad_w, pad_h)


class YOLOOBBDataset(BaseYOLODataset):
    """Dataset for YOLO OBB format (4-corner annotations)."""

    def _load_labels(self, path: Path) -> torch.Tensor:
        """Load OBB labels: class + 8 corner coordinates -> (class, cx, cy, w, h, angle)."""
        if not path.exists():
            return torch.zeros((0, 6), dtype=torch.float32)

        labels = []
        has_standard_detection = False
        with open(path) as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                # Standard detection format: class x y w h (no rotation)
                has_standard_detection = True
                try:
                    cls = int(parts[0])
                    if not (0 <= cls < self.num_classes):
                        continue
                    x, y, w, h = map(float, parts[1:5])
                    # Append with rotation = 0
                    labels.append([cls, x, y, w, h, 0.0])
                except ValueError as e:
                    logger.debug(f"Skipping invalid line in {path}: {e}")
                    continue
            elif len(parts) == 9:
                # OBB format: class + 8 corner coordinates
                try:
                    cls = int(parts[0])
                    if not (0 <= cls < self.num_classes):
                        continue
                    corners = np.array([float(x) for x in parts[1:9]], dtype=np.float32).reshape(4, 2)
                    labels.append([cls, *corners_to_xywhr(corners)])
                except ValueError as e:
                    logger.debug(f"Skipping invalid line in {path}: {e}")
                    continue
            else:
                # Skip invalid formats
                if parts:  # Only warn if line is not empty
                    warnings.warn(
                        f"Unsupported format in {path}: expected 5 values (standard detection) "
                        f"or 9 values (OBB), got {len(parts)} values",
                        UserWarning,
                        stacklevel=2,
                    )
                continue

        # Warn only once per dataset to avoid log noise
        if has_standard_detection and labels and not self._standard_format_warned:
            warnings.warn(
                "Standard detection format detected. "
                "Using axis-aligned bounding boxes with rotation set to 0. "
                "For optimal OBB training, please provide annotations in OBB format (8 corner coordinates).",
                UserWarning,
                stacklevel=2,
            )
            self._standard_format_warned = True

        return torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 6), dtype=torch.float32)


class YOLODetDataset(BaseYOLODataset):
    """Dataset for standard YOLO detection format (axis-aligned bounding boxes)."""

    def _load_labels(self, path: Path) -> torch.Tensor:
        """Load standard YOLO format: class x_center y_center width height (normalized)."""
        if not path.exists():
            return torch.zeros((0, 5), dtype=torch.float32)

        labels = []
        with open(path) as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            # Standard YOLO format: class cx cy w h (5 parts)
            if len(parts) != 5:
                continue
            try:
                cls = int(parts[0])
                if not (0 <= cls < self.num_classes):
                    continue
                bbox = [float(x) for x in parts[1:5]]
                # Check that coordinates are already normalized (all values in [0.0, 1.0])
                if not all(0.0 <= val <= 1.0 for val in bbox):
                    logger.warning(
                        f"Skipping label in {path}: coordinates appear to be out of normalized range [0.0, 1.0]"
                    )
                    continue
                labels.append([cls, *bbox])
            except ValueError:
                continue

        return torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5), dtype=torch.float32)


# =============================================================================
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
        num_objects: int = 3,
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
            num_objects: Number of objects to place in each image.
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
            ...     # Check structure
            ...     (dataset_path / "images" / "train").exists()
            True
            ...     (dataset_path / "labels" / "train").exists()
            True
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
                    num_objects=num_objects,
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
