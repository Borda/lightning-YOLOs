"""
Dataset classes for YOLO.
"""

import logging
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from lit_yolo.data.utils import corners_to_xywhr

logger = logging.getLogger(__name__)


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

    @staticmethod
    def _parse_standard_detection_line(parts: list[str], num_classes: int) -> list[float] | None:
        """Parse standard detection format line: class x y w h (no rotation).
        
        Args:
            parts: Line parts split by whitespace.
            num_classes: Number of classes for validation.
            
        Returns:
            List of [cls, x, y, w, h, 0.0] or None if invalid.
        """
        try:
            cls = int(parts[0])
            if not (0 <= cls < num_classes):
                return None
            x, y, w, h = map(float, parts[1:5])
            # Append with rotation = 0
            return [cls, x, y, w, h, 0.0]
        except ValueError as e:
            logger.debug(f"Skipping invalid standard detection line: {e}")
            return None

    @staticmethod
    def _parse_obb_line(parts: list[str], num_classes: int) -> list[float] | None:
        """Parse OBB format line: class + 8 corner coordinates.
        
        Args:
            parts: Line parts split by whitespace.
            num_classes: Number of classes for validation.
            
        Returns:
            List of [cls, cx, cy, w, h, angle] or None if invalid.
        """
        try:
            cls = int(parts[0])
            if not (0 <= cls < num_classes):
                return None
            corners = np.array([float(x) for x in parts[1:9]], dtype=np.float32).reshape(4, 2)
            return [cls, *corners_to_xywhr(corners)]
        except ValueError as e:
            logger.debug(f"Skipping invalid OBB line: {e}")
            return None

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
                label = self._parse_standard_detection_line(parts, self.num_classes)
                if label is not None:
                    has_standard_detection = True
                    labels.append(label)
            elif len(parts) == 9:
                # OBB format: class + 8 corner coordinates
                label = self._parse_obb_line(parts, self.num_classes)
                if label is not None:
                    labels.append(label)
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
