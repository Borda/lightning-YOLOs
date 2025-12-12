from typing import Any

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Default class colors for visualization (BGR format for OpenCV)
DEFAULT_COLORS = [
    (128, 0, 0),  # Dark Blue
    (0, 128, 0),  # Dark Green
    (0, 0, 128),  # Dark Red
    (128, 128, 0),  # Olive (Dark Green + Dark Blue)
    (255, 255, 0),  # Cyan (Green + Blue)
    (255, 0, 255),  # Magenta (Blue + Red)
    (0, 255, 255),  # Yellow (Green + Red)
    # (255, 0, 0),  # Blue
    # (0, 255, 0),  # Green
    # (0, 0, 255),  # Red
]


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
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)
        label_y1 = max(y1, label_h + 10)

        # Draw label background
        cv2.rectangle(img_draw, (x1, label_y1 - label_h - 10), (x1 + label_w, label_y1), color, -1)

        # Draw label text
        cv2.putText(img_draw, label, (x1, label_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

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
        corners = np.array(
            [
                [cx_px + hw * cos_a - hh * sin_a, cy_px + hw * sin_a + hh * cos_a],
                [cx_px + hw * cos_a + hh * sin_a, cy_px + hw * sin_a - hh * cos_a],
                [cx_px - hw * cos_a + hh * sin_a, cy_px - hw * sin_a - hh * cos_a],
                [cx_px - hw * cos_a - hh * sin_a, cy_px - hw * sin_a + hh * cos_a],
            ],
            dtype=np.int32,
        )

        # Draw rotated rectangle
        cv2.polylines(img_draw, [corners], True, color, line_thickness)

        # Draw label at top-left corner
        label = f"{int(cls_id)}"
        if class_names is not None and 0 <= int(cls_id) < len(class_names):
            label = f"{class_names[int(cls_id)]}: {int(cls_id)}"

        # Find the top-left corner for label placement
        x1, y1 = int(corners[:, 0].min()), int(corners[:, 1].min())

        # Calculate label size and position
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        label_y1 = max(y1, label_h + 10)

        # Draw label background
        cv2.rectangle(img_draw, (x1, label_y1 - label_h - 10), (x1 + label_w, label_y1), color, -1)

        # Draw label text
        cv2.putText(img_draw, label, (x1, label_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return img_draw


def annotate_batch_images(
    batch: dict,
    draw_fn: callable,
    class_names: list[str] | None = None,
) -> list[np.ndarray]:
    """Annotate images in a batch with bounding boxes.

    Args:
        batch: Batch dictionary containing images, bboxes, class ids, and batch indices.
               Expected keys: "img" (B, 3, H, W), "batch_idx" (N,), "cls" (N, 1), "bboxes" (N, 4/5).
        draw_fn: Function to draw boxes on image. Should have signature:
                 (img: np.ndarray, bboxes: np.ndarray, class_ids: np.ndarray, class_names: list[str] | None) -> np.ndarray
        class_names: Optional list of class names for labels.

    Returns:
        List of annotated images in BGR format.
    """
    # Extract data from batch
    imgs = batch["img"]  # (B, 3, H, W) in [0, 1]
    batch_indices = batch["batch_idx"]  # (N,)
    cls = batch["cls"]  # (N, 1)
    bboxes = batch["bboxes"]  # (N, 4 or 5) depending on format

    # Convert tensors to numpy
    imgs_np = imgs.cpu().numpy()
    batch_indices_np = batch_indices.cpu().numpy()
    cls_np = cls.cpu().numpy().flatten()
    bboxes_np = bboxes.cpu().numpy()

    # Create annotated images
    annotated_images = []
    for b in range(imgs_np.shape[0]):
        # Get image and convert from CHW to HWC, scale to [0, 255]
        img = (imgs_np[b].transpose(1, 2, 0) * 255).astype(np.uint8)

        # Get boxes for this image
        mask = batch_indices_np == b
        img_bboxes = bboxes_np[mask]
        img_cls = cls_np[mask]

        if len(img_bboxes) > 0:
            # Draw bounding boxes using provided drawing function
            img = draw_fn(img, img_bboxes, img_cls, class_names=class_names)
        else:
            # Convert RGB to BGR if no boxes to draw
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        annotated_images.append(img)

    return annotated_images


def show_images_in_grid(
    images: list[np.ndarray],
) -> tuple[Any, np.ndarray]:
    """Create a grid visualization of images using matplotlib subplots.

    Args:
        images: List of images in BGR format (H, W, 3).

    Returns:
        Tuple of (figure, axes) from matplotlib.
        Caller is responsible for saving/showing and closing the figure.
    """
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

    return fig, axes
