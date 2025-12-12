"""
Utility functions for YOLO data processing and synthetic dataset generation.
"""

import logging
import math
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Default class colors for visualization (BGR format for OpenCV)
DEFAULT_COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 0, 0),      # Dark Red
    (0, 128, 0),      # Dark Green
    (0, 0, 128),      # Dark Blue
    (128, 128, 0),    # Olive
]


def read_class_names_from_yaml(data_path: str | Path) -> list[str] | None:
    """Read class names from dataset YAML file.

    Looks for data.yaml or dataset.yaml in the dataset directory and extracts class names.

    Args:
        data_path: Path to dataset root directory.

    Returns:
        List of class names if found in YAML file, None otherwise.

    Examples:
        >>> from pathlib import Path
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     yaml_path = Path(tmpdir) / "data.yaml"
        ...     yaml_path.write_text("names: ['cat', 'dog', 'bird']")
        ...     names = read_class_names_from_yaml(tmpdir)
        >>> names is not None
        True
    """
    try:
        import yaml
    except ImportError:
        logger.debug("PyYAML not installed, cannot read class names from YAML file")
        return None

    data_path = Path(data_path)

    # Try common YAML file names
    for yaml_name in ["data.yaml", "dataset.yaml", "data.yml", "dataset.yml"]:
        yaml_file = data_path / yaml_name
        if yaml_file.exists():
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                    if data and "names" in data:
                        names = data["names"]
                        # Handle both list and dict formats
                        if isinstance(names, dict):
                            # Convert dict to list sorted by key
                            names = [names[i] for i in sorted(names.keys())]
                        logger.debug(f"Loaded {len(names)} class names from {yaml_file}")
                        return names
            except Exception as e:
                logger.debug(f"Failed to read class names from {yaml_file}: {e}")
                continue

    logger.debug(f"No class names found in YAML files at {data_path}")
    return None


# Available shapes for synthetic dataset generation
SYNTHETIC_SHAPES = ["square", "triangle", "circle"]

# Available colors for synthetic dataset generation (RGB format)
SYNTHETIC_COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
}


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
    min_objects: int,
    max_objects: int,
    class_mode: Literal["shape", "color"],
    min_size_ratio: float,
    max_size_ratio: float,
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    """Generate a single synthetic image with labeled objects.

    Args:
        img_size: Size of the image (square).
        min_objects: Minimum number of objects to generate.
        max_objects: Maximum number of objects to generate.
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

    # Randomly decide how many objects to generate
    num_objects = np.random.randint(min_objects, max_objects + 1)

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
        margin = min(obj_size, (img_size // 2) - 1)
        # Ensure the range is valid for randint; if not, place at center
        if margin < img_size - margin:
            cx = np.random.randint(margin, img_size - margin)
            cy = np.random.randint(margin, img_size - margin)
        else:
            cx = img_size // 2
            cy = img_size // 2

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


def draw_bboxes_on_image(
    img: np.ndarray,
    bboxes: np.ndarray,
    class_ids: np.ndarray,
    class_names: list[str] | None = None,
    colors: list[tuple[int, int, int]] | None = None,
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw axis-aligned bounding boxes on image.

    Args:
        img: Image array (H, W, 3) in RGB format with values in [0, 255].
        bboxes: Array of shape (N, 4) with [cx, cy, w, h] in normalized coordinates [0, 1].
        class_ids: Array of shape (N,) with class indices.
        class_names: Optional list of class names for labels.
        colors: Optional list of BGR color tuples for each class.
        line_thickness: Thickness of bounding box lines.

    Returns:
        Image with drawn bounding boxes (BGR format for OpenCV display).

    Examples:
        >>> import numpy as np
        >>> img = np.ones((640, 640, 3), dtype=np.uint8) * 128
        >>> bboxes = np.array([[0.5, 0.5, 0.2, 0.3]])
        >>> class_ids = np.array([0])
        >>> result = draw_bboxes_on_image(img, bboxes, class_ids)
        >>> result.shape
        (640, 640, 3)
    """
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError(f"Image must have shape (H, W, 3), got {img.shape}")

    # Convert RGB to BGR for OpenCV
    img_draw = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
    h, w = img_draw.shape[:2]

    if colors is None:
        colors = DEFAULT_COLORS

    for i, (bbox, cls_id) in enumerate(zip(bboxes, class_ids)):
        cx, cy, box_w, box_h = bbox
        # Convert normalized coordinates to pixel coordinates
        x1 = int((cx - box_w / 2) * w)
        y1 = int((cy - box_h / 2) * h)
        x2 = int((cx + box_w / 2) * w)
        y2 = int((cy + box_h / 2) * h)

        # Get color for this class
        color = colors[int(cls_id) % len(colors)]

        # Draw rectangle
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, line_thickness)

        # Draw label
        label = f"{int(cls_id)}"
        if class_names is not None and 0 <= int(cls_id) < len(class_names):
            label = f"{class_names[int(cls_id)]}: {int(cls_id)}"

        # Calculate label size and position
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y1 = max(y1, label_h + 10)

        # Draw label background
        cv2.rectangle(img_draw, (x1, label_y1 - label_h - 10), (x1 + label_w, label_y1), color, -1)

        # Draw label text
        cv2.putText(img_draw, label, (x1, label_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img_draw


def draw_obb_on_image(
    img: np.ndarray,
    obbs: np.ndarray,
    class_ids: np.ndarray,
    class_names: list[str] | None = None,
    colors: list[tuple[int, int, int]] | None = None,
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw oriented bounding boxes on image.

    Args:
        img: Image array (H, W, 3) in RGB format with values in [0, 255].
        obbs: Array of shape (N, 5) with [cx, cy, w, h, angle] in normalized coordinates.
        class_ids: Array of shape (N,) with class indices.
        class_names: Optional list of class names for labels.
        colors: Optional list of BGR color tuples for each class.
        line_thickness: Thickness of bounding box lines.

    Returns:
        Image with drawn oriented bounding boxes (BGR format for OpenCV display).

    Examples:
        >>> import numpy as np
        >>> img = np.ones((640, 640, 3), dtype=np.uint8) * 128
        >>> obbs = np.array([[0.5, 0.5, 0.2, 0.3, 0.0]])
        >>> class_ids = np.array([0])
        >>> result = draw_obb_on_image(img, obbs, class_ids)
        >>> result.shape
        (640, 640, 3)
    """
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError(f"Image must have shape (H, W, 3), got {img.shape}")

    # Convert RGB to BGR for OpenCV
    img_draw = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
    h, w = img_draw.shape[:2]

    if colors is None:
        colors = DEFAULT_COLORS

    for i, (obb, cls_id) in enumerate(zip(obbs, class_ids)):
        cx, cy, box_w, box_h, angle = obb

        # Convert normalized coordinates to pixel coordinates
        cx_px = cx * w
        cy_px = cy * h
        w_px = box_w * w
        h_px = box_h * h

        # Get color for this class
        color = colors[int(cls_id) % len(colors)]

        # Calculate rotated rectangle corners
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        hw, hh = w_px / 2, h_px / 2

        # Calculate the four corners
        corners = np.array([
            [cx_px + hw * cos_a - hh * sin_a, cy_px + hw * sin_a + hh * cos_a],
            [cx_px + hw * cos_a + hh * sin_a, cy_px + hw * sin_a - hh * cos_a],
            [cx_px - hw * cos_a + hh * sin_a, cy_px - hw * sin_a - hh * cos_a],
            [cx_px - hw * cos_a - hh * sin_a, cy_px - hw * sin_a + hh * cos_a],
        ], dtype=np.int32)

        # Draw rotated rectangle
        cv2.polylines(img_draw, [corners], True, color, line_thickness)

        # Draw label at top-left corner
        label = f"{int(cls_id)}"
        if class_names is not None and 0 <= int(cls_id) < len(class_names):
            label = f"{class_names[int(cls_id)]}: {int(cls_id)}"

        # Find the top-left corner for label placement
        x1, y1 = int(corners[:, 0].min()), int(corners[:, 1].min())

        # Calculate label size and position
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y1 = max(y1, label_h + 10)

        # Draw label background
        cv2.rectangle(img_draw, (x1, label_y1 - label_h - 10), (x1 + label_w, label_y1), color, -1)

        # Draw label text
        cv2.putText(img_draw, label, (x1, label_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img_draw


