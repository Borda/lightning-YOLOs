"""Shared test fixtures for unit tests."""

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture(scope="module")
def create_test_image():
    """Factory fixture to create test images."""

    def _create_image(img_path: Path, size: int = 640):
        """Create a dummy test image at the specified path."""
        img = np.ones((size, size, 3), dtype=np.uint8) * 128
        cv2.imwrite(str(img_path), img)

    return _create_image


@pytest.fixture
def yolo_dataset_dir(tmp_path, create_test_image):
    """Create a reusable YOLO dataset directory structure.

    This fixture creates a temporary dataset directory with the standard YOLO structure:
    - images/train/ (for training images)
    - labels/train/ (for label files, empty to be populated by individual tests)

    Can be used for any YOLO format (standard detection, OBB, etc.).
    """
    root = tmp_path / "yolo_dataset"
    # Create directory structure
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)

    return root


@pytest.fixture
def bounding_box_dataset(yolo_dataset_dir, create_test_image):
    """Create a synthetic dataset with standard detection format (5 values).

    Returns:
        Path to dataset root with a single image and standard detection labels.
    """
    # Create test image
    img_path = yolo_dataset_dir / "images" / "train" / "test1.jpg"
    create_test_image(img_path)

    # Create label file with standard detection format
    label_path = yolo_dataset_dir / "labels" / "train" / "test1.txt"
    with open(label_path, "w") as f:
        f.write("0 0.5 0.5 0.3 0.4\n")
        f.write("1 0.3 0.3 0.2 0.2\n")

    return yolo_dataset_dir


@pytest.fixture
def oriented_bounding_box_dataset(yolo_dataset_dir, create_test_image):
    """Create a synthetic OBB dataset with oriented bounding box format.

    Creates a dataset with OBB format labels (9 values: class + 8 corner coordinates).

    Returns:
        Path to dataset root with a single image and OBB format labels.
    """
    # Create test image
    img_path = yolo_dataset_dir / "images" / "train" / "test2.jpg"
    create_test_image(img_path)

    # Create label file with OBB format (rectangle corners)
    label_path = yolo_dataset_dir / "labels" / "train" / "test2.txt"
    with open(label_path, "w") as f:
        # Simple axis-aligned rectangle
        f.write("0 0.3 0.3 0.7 0.3 0.7 0.7 0.3 0.7\n")

    return yolo_dataset_dir


@pytest.fixture
def mixed_detection_dataset(yolo_dataset_dir, create_test_image):
    """Create a synthetic dataset with mixed plain bounding boxes and OBB formats.

    Creates a dataset containing:
    - One image with standard detection format (5 values: class x y w h)
    - One image with OBB format (9 values: class + 8 corner coordinates)

    Returns:
        Path to dataset root with two images using different annotation formats.
    """
    # Create test images
    img1_path = yolo_dataset_dir / "images" / "train" / "test_bb.jpg"
    img2_path = yolo_dataset_dir / "images" / "train" / "test_obb.jpg"
    create_test_image(img1_path)
    create_test_image(img2_path)

    # Standard detection format
    label1_path = yolo_dataset_dir / "labels" / "train" / "test_bb.txt"
    with open(label1_path, "w") as f:
        f.write("0 0.5 0.5 0.3 0.4\n")

    # OBB format
    label2_path = yolo_dataset_dir / "labels" / "train" / "test_obb.txt"
    with open(label2_path, "w") as f:
        f.write("0 0.3 0.3 0.7 0.3 0.7 0.7 0.3 0.7\n")

    return yolo_dataset_dir
