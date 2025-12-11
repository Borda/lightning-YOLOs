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
def obb_dataset_dir(tmp_path, create_test_image):
    """Create a reusable OBB dataset directory structure with sample data.

    This fixture creates a temporary dataset with the following structure:
    - images/train/ with test images
    - labels/train/ (empty, to be populated by individual tests)
    """
    root = tmp_path / "obb_dataset"
    # Create directory structure
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)

    return root


@pytest.fixture
def standard_detection_dataset(obb_dataset_dir, create_test_image):
    """Create a synthetic dataset with standard detection format (5 values).

    Returns:
        Path to dataset root with a single image and standard detection labels.
    """
    # Create test image
    img_path = obb_dataset_dir / "images" / "train" / "test1.jpg"
    create_test_image(img_path)

    # Create label file with standard detection format
    label_path = obb_dataset_dir / "labels" / "train" / "test1.txt"
    with open(label_path, "w") as f:
        f.write("0 0.5 0.5 0.3 0.4\n")
        f.write("1 0.3 0.3 0.2 0.2\n")

    return obb_dataset_dir


@pytest.fixture
def obb_format_dataset(obb_dataset_dir, create_test_image):
    """Create a synthetic dataset with OBB format (9 values).

    Returns:
        Path to dataset root with a single image and OBB format labels.
    """
    # Create test image
    img_path = obb_dataset_dir / "images" / "train" / "test2.jpg"
    create_test_image(img_path)

    # Create label file with OBB format (rectangle corners)
    label_path = obb_dataset_dir / "labels" / "train" / "test2.txt"
    with open(label_path, "w") as f:
        # Simple axis-aligned rectangle
        f.write("0 0.3 0.3 0.7 0.3 0.7 0.7 0.3 0.7\n")

    return obb_dataset_dir


@pytest.fixture
def mixed_format_dataset(obb_dataset_dir, create_test_image):
    """Create a synthetic dataset with mixed standard detection and OBB formats.

    Returns:
        Path to dataset root with two images - one with standard detection, one with OBB format.
    """
    # Create test images
    img1_path = obb_dataset_dir / "images" / "train" / "test_standard.jpg"
    img2_path = obb_dataset_dir / "images" / "train" / "test_obb.jpg"
    create_test_image(img1_path)
    create_test_image(img2_path)

    # Standard detection format
    label1_path = obb_dataset_dir / "labels" / "train" / "test_standard.txt"
    with open(label1_path, "w") as f:
        f.write("0 0.5 0.5 0.3 0.4\n")

    # OBB format
    label2_path = obb_dataset_dir / "labels" / "train" / "test_obb.txt"
    with open(label2_path, "w") as f:
        f.write("0 0.3 0.3 0.7 0.3 0.7 0.7 0.3 0.7\n")

    return obb_dataset_dir
