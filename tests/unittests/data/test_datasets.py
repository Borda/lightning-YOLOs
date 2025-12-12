"""Unit tests for synthetic dataset creation."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from lit_yolo import OBBDataset
from lit_yolo.data import BaseDataModule, DetDataModule, OBBDataModule
from lit_yolo.data.utils import calculate_bbox_iou, calculate_boundary_overlap, generate_synthetic_sample


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for deterministic results."""
    np.random.seed(42)
    yield


class TestOverlapHelperFunctions:
    """Tests for overlap calculation helper functions."""

    def test_calculate_bbox_iou_non_overlapping(self):
        """Test IoU calculation for non-overlapping boxes."""
        bbox1 = (0.25, 0.25, 0.2, 0.2)
        bbox2 = (0.75, 0.75, 0.2, 0.2)
        iou = calculate_bbox_iou(bbox1, bbox2)
        assert iou == 0.0, f"Non-overlapping boxes should have IoU of 0.0, got {iou}"

    def test_calculate_bbox_iou_identical(self):
        """Test IoU calculation for identical boxes."""
        bbox1 = (0.5, 0.5, 0.2, 0.2)
        bbox2 = (0.5, 0.5, 0.2, 0.2)
        iou = calculate_bbox_iou(bbox1, bbox2)
        assert round(iou, 10) == 1.0, f"Identical boxes should have IoU of 1.0, got {iou}"

    def test_calculate_bbox_iou_partial_overlap(self):
        """Test IoU calculation for partially overlapping boxes."""
        bbox1 = (0.5, 0.5, 0.4, 0.4)
        bbox2 = (0.6, 0.6, 0.4, 0.4)
        iou = calculate_bbox_iou(bbox1, bbox2)
        # Expected: 0.09 / 0.23 = 9/23 ≈ 0.391304347826087
        expected_iou = 9 / 23
        assert abs(iou - expected_iou) < 1e-10, f"Partially overlapping boxes should have IoU of {expected_iou:.10f}, got {iou}"

    def test_calculate_boundary_overlap_inside(self):
        """Test boundary overlap for box fully inside image."""
        bbox = (0.5, 0.5, 0.2, 0.2)
        overlap = calculate_boundary_overlap(bbox)
        assert overlap == 0.0, f"Box fully inside should have boundary overlap of 0.0, got {overlap}"

    def test_calculate_boundary_overlap_at_edge(self):
        """Test boundary overlap for box at image edge."""
        bbox = (0.05, 0.5, 0.2, 0.2)  # Box extends beyond left edge
        overlap = calculate_boundary_overlap(bbox)
        assert overlap == 0.25, f"Box at edge should have boundary overlap of 0.25, got {overlap}"

    def test_calculate_boundary_overlap_at_corner(self):
        """Test boundary overlap for box at image corner."""
        bbox = (0.05, 0.05, 0.2, 0.2)  # Box at top-left corner
        overlap = calculate_boundary_overlap(bbox)
        assert overlap == 0.4375, f"Box at corner should have overlap of 0.4375, got {overlap}"

    def test_calculate_boundary_overlap_fully_outside(self):
        """Test boundary overlap for box fully outside image."""
        bbox = (-0.5, -0.5, 0.2, 0.2)  # Box completely outside
        overlap = calculate_boundary_overlap(bbox)
        assert overlap == 1.0, f"Box fully outside should have boundary overlap of 1.0, got {overlap}"


class TestOverlapThresholdInGeneration:
    """Tests for overlap threshold in synthetic sample generation."""

    def test_generate_with_low_overlap_threshold(self):
        """Test that low overlap threshold prevents objects from overlapping significantly."""
        img, labels = generate_synthetic_sample(
            img_size=640,
            min_objects=5,
            max_objects=5,
            class_mode="shape",
            min_size_ratio=0.15,
            max_size_ratio=0.25,
            overlap_threshold=0.1,  # Very low threshold
        )

        # Check that we got some objects (might not be all 5 due to placement difficulty)
        assert len(labels) > 0, "Should generate at least some objects"

        # Check that no two objects overlap significantly
        for i, (cls1, cx1, cy1, w1, h1) in enumerate(labels):
            for j, (cls2, cx2, cy2, w2, h2) in enumerate(labels):
                if i >= j:
                    continue
                iou = calculate_bbox_iou((cx1, cy1, w1, h1), (cx2, cy2, w2, h2))
                assert iou <= 0.1 + 1e-9, f"Objects {i} and {j} overlap with IoU {iou:.3f}, exceeds threshold 0.1"

    def test_generate_with_high_overlap_threshold(self):
        """Test that high overlap threshold allows more objects to be placed."""
        img, labels = generate_synthetic_sample(
            img_size=640,
            min_objects=5,
            max_objects=5,
            class_mode="shape",
            min_size_ratio=0.15,
            max_size_ratio=0.25,
            overlap_threshold=0.5,  # High threshold allows more overlap
        )

        # With higher threshold, we should be able to place more objects
        assert len(labels) >= 3, "Should generate at least 3 objects with high overlap threshold"

    def test_objects_within_boundaries(self):
        """Test that objects respect boundary overlap threshold."""
        img, labels = generate_synthetic_sample(
            img_size=640,
            min_objects=5,
            max_objects=5,
            class_mode="shape",
            min_size_ratio=0.1,
            max_size_ratio=0.2,
            overlap_threshold=0.2,
        )

        # Check that all objects respect boundary overlap threshold
        for cls, cx, cy, w, h in labels:
            boundary_overlap = calculate_boundary_overlap((cx, cy, w, h))
            assert boundary_overlap <= 0.2 + 1e-9, (
                f"Object at ({cx:.3f}, {cy:.3f}) with size ({w:.3f}, {h:.3f}) "
                f"has boundary overlap {boundary_overlap:.3f}, exceeds threshold 0.2"
            )

    def test_generate_with_zero_overlap_threshold(self):
        """Test that zero overlap threshold prevents any overlap."""
        img, labels = generate_synthetic_sample(
            img_size=640,
            min_objects=3,
            max_objects=3,
            class_mode="shape",
            min_size_ratio=0.1,
            max_size_ratio=0.15,
            overlap_threshold=0.0,  # No overlap allowed
        )

        # With zero overlap, might not place all objects
        assert len(labels) >= 1, "Should place at least one object"

        # Verify no overlap between objects
        for i, (cls1, cx1, cy1, w1, h1) in enumerate(labels):
            for j, (cls2, cx2, cy2, w2, h2) in enumerate(labels):
                if i >= j:
                    continue
                iou = calculate_bbox_iou((cx1, cy1, w1, h1), (cx2, cy2, w2, h2))
                assert iou < 1e-9, f"Objects {i} and {j} should not overlap with zero threshold, got IoU {iou}"

            # Verify no boundary overlap
            boundary_overlap = calculate_boundary_overlap((cx1, cy1, w1, h1))
            assert boundary_overlap < 1e-9, f"Object {i} should not overlap with boundaries, got {boundary_overlap}"

    def test_default_overlap_threshold(self):
        """Test that default overlap threshold (0.3) works correctly."""
        img, labels = generate_synthetic_sample(
            img_size=640,
            min_objects=4,
            max_objects=4,
            class_mode="shape",
            min_size_ratio=0.1,
            max_size_ratio=0.2,
            # Using default overlap_threshold=0.3
        )

        assert len(labels) > 0, "Should generate at least some objects with default threshold"

        # Verify overlaps respect default threshold
        for i, (cls1, cx1, cy1, w1, h1) in enumerate(labels):
            for j, (cls2, cx2, cy2, w2, h2) in enumerate(labels):
                if i >= j:
                    continue
                iou = calculate_bbox_iou((cx1, cy1, w1, h1), (cx2, cy2, w2, h2))
                assert iou <= 0.3 + 1e-9, f"Objects {i} and {j} overlap exceeds default threshold, got IoU {iou}"


class TestSyntheticDatasetCreation:
    """Tests for synthetic dataset creation function."""

    def test_basic_creation(self, tmp_path):
        """Test basic synthetic dataset creation."""
        dataset_path = BaseDataModule.create_synthetic_dataset(
            tmp_path, num_samples=7, split_ratio=0.7, img_size=640, class_mode="shape"
        )

        # Verify path is returned correctly
        assert dataset_path == tmp_path

        # Verify directory structure
        assert (dataset_path / "images" / "train").exists()
        assert (dataset_path / "images" / "val").exists()
        assert (dataset_path / "labels" / "train").exists()
        assert (dataset_path / "labels" / "val").exists()

        # Verify number of images (70% train, 30% val of 7 samples = 4 train, 3 val)
        train_images = list((dataset_path / "images" / "train").glob("*.jpg"))
        val_images = list((dataset_path / "images" / "val").glob("*.jpg"))
        assert len(train_images) == 4
        assert len(val_images) == 3

        # Verify number of label files
        train_labels = list((dataset_path / "labels" / "train").glob("*.txt"))
        val_labels = list((dataset_path / "labels" / "val").glob("*.txt"))
        assert len(train_labels) == 4
        assert len(val_labels) == 3


    @pytest.mark.parametrize("class_mode", ["shape", "color"])
    def test_class_modes(self, tmp_path, class_mode):
        """Test synthetic dataset with different classification modes."""
        dataset_path = BaseDataModule.create_synthetic_dataset(
            tmp_path, num_samples=15, split_ratio=0.67, class_mode=class_mode
        )

        # Check that labels contain only class indices 0, 1, 2
        label_files = list((dataset_path / "labels" / "train").glob("*.txt"))
        classes_found = set()

        for label_file in label_files:
            with open(label_file) as f:
                lines = [ln.strip() for ln in f.readlines()]
                for line in lines:
                    if line:  # Skip empty lines
                        parts = line.split()
                        cls = int(parts[0])
                        classes_found.add(cls)

        # Should have 3 classes (0, 1, 2)
        assert classes_found == {0, 1, 2}


    def test_invalid_class_mode(self, tmp_path):
        """Test that invalid class_mode raises ValueError."""
        with pytest.raises(ValueError, match="class_mode"):
            BaseDataModule.create_synthetic_dataset(tmp_path, class_mode="invalid")


    def test_label_format_validity(self, tmp_path):
        """Test that generated labels are in valid YOLO format."""
        # Use fixed min/max to have predictable object counts
        dataset_path = BaseDataModule.create_synthetic_dataset(
            tmp_path, num_samples=7, split_ratio=0.7, min_objects=3, max_objects=3
        )

        # Check label format (class cx cy w h, all normalized)
        label_files = list((dataset_path / "labels" / "train").glob("*.txt"))

        for label_file in label_files:
            with open(label_file) as f:
                lines = [ln.strip() for ln in f.readlines()]
                # Filter out empty lines
            lines = [ln for ln in lines if ln]
            # Each image should have 3 objects (min=max=3)
            assert len(lines) == 3

            for line in lines:
                parts = line.split()
                # Should have 5 parts: class, cx, cy, w, h
                assert len(parts) == 5

                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])

                # Class should be 0, 1, or 2
                assert cls in [0, 1, 2]

                # Coordinates should be normalized (0-1 range)
                assert 0.0 <= cx <= 1.0
                assert 0.0 <= cy <= 1.0
                assert 0.0 < w <= 1.0
                assert 0.0 < h <= 1.0


    def test_image_properties(self, tmp_path):
        """Test that generated images have correct properties."""
        img_size = 512
        dataset_path = BaseDataModule.create_synthetic_dataset(
            tmp_path, num_samples=4, split_ratio=0.75, img_size=img_size
        )

        # Check image properties
        img_files = list((dataset_path / "images" / "train").glob("*.jpg"))

        for img_file in img_files:
            img = cv2.imread(str(img_file))
            assert img is not None
            # Check image size
            assert img.shape[0] == img_size
            assert img.shape[1] == img_size
            # Check image has 3 channels (RGB)
            assert img.shape[2] == 3


    def test_image_label_correspondence(self, tmp_path):
        """Test that each image has a corresponding label file."""
        dataset_path = BaseDataModule.create_synthetic_dataset(tmp_path, num_samples=15, split_ratio=0.67)

        for split in ["train", "val"]:
            img_dir = dataset_path / "images" / split
            label_dir = dataset_path / "labels" / split

            img_files = sorted(img_dir.glob("*.jpg"))
            label_files = sorted(label_dir.glob("*.txt"))

            # Same number of files
            assert len(img_files) == len(label_files)

            # Matching filenames (stem)
            for img_file, label_file in zip(img_files, label_files):
                assert img_file.stem == label_file.stem


    def test_reproducibility_with_seed(self, tmp_path):
        """Test that same seed produces same dataset."""
        # Create two separate temp directories for comparison
        root1 = tmp_path / "dataset1"
        root2 = tmp_path / "dataset2"

        # Create two datasets with same seed
        BaseDataModule.create_synthetic_dataset(root1, num_samples=7, split_ratio=0.7, seed=123)
        BaseDataModule.create_synthetic_dataset(root2, num_samples=7, split_ratio=0.7, seed=123)

        # Compare labels (should be identical)
        label_files1 = sorted((root1 / "labels" / "train").glob("*.txt"))
        label_files2 = sorted((root2 / "labels" / "train").glob("*.txt"))

        for lf1, lf2 in zip(label_files1, label_files2):
            with open(lf1) as f1, open(lf2) as f2:
                content1 = f1.read()
                content2 = f2.read()
                # Labels should be identical
                assert content1 == content2


    def test_custom_image_size(self, tmp_path):
        """Test synthetic dataset creation with custom image size."""
        custom_size = 320
        dataset_path = BaseDataModule.create_synthetic_dataset(
            tmp_path, num_samples=4, split_ratio=0.75, img_size=custom_size
        )

        # Check that images have the custom size
        img_files = list((dataset_path / "images" / "train").glob("*.jpg"))
        for img_file in img_files:
            img = cv2.imread(str(img_file))
            assert img.shape[0] == custom_size
            assert img.shape[1] == custom_size


    def test_can_load_with_det_datamodule(self, tmp_path):
        """Test that synthetic dataset can be loaded with DetDataModule."""
        dataset_path = BaseDataModule.create_synthetic_dataset(tmp_path, num_samples=15, split_ratio=0.67)

        # Create DetDataModule and load dataset
        datamodule = DetDataModule(data=str(dataset_path), img_size=640, batch_size=2, num_classes=3)
        datamodule.setup("fit")

        # Verify datasets are created
        assert datamodule.train_ds is not None
        assert datamodule.val_ds is not None
        assert len(datamodule.train_ds) == 10
        assert len(datamodule.val_ds) == 5

        # Verify can get batch
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))

        # Check batch structure
        assert "img" in batch
        assert "cls" in batch
        assert "bboxes" in batch
        assert batch["img"].shape[0] == 2  # batch size
        assert batch["img"].shape[1] == 3  # channels
        assert batch["bboxes"].shape[1] == 4  # cx, cy, w, h


    def test_can_load_with_obb_datamodule(self, tmp_path):
        """Test that synthetic dataset can be loaded with OBBDataModule (standard format)."""
        dataset_path = BaseDataModule.create_synthetic_dataset(tmp_path, num_samples=15, split_ratio=0.67)

        # Create OBBDataModule and load dataset
        # OBB module should also handle standard detection format
        datamodule = OBBDataModule(data=str(dataset_path), img_size=640, batch_size=2, num_classes=3)
        datamodule.setup("fit")

        # Verify datasets are created
        assert datamodule.train_ds is not None
        assert datamodule.val_ds is not None
        assert len(datamodule.train_ds) == 10
        assert len(datamodule.val_ds) == 5

        # Verify can get batch
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))

        # Check batch structure
        assert "img" in batch
        assert "cls" in batch
        assert "bboxes" in batch
        assert batch["img"].shape[0] == 2  # batch size
        assert batch["bboxes"].shape[1] == 5  # cx, cy, w, h, angle


    def test_string_path_input(self, tmp_path):
        """Test that method accepts string path as input."""
        root_str = str(tmp_path)
        dataset_path = BaseDataModule.create_synthetic_dataset(root_str, num_samples=4, split_ratio=0.75)

        # Should return Path object
        assert isinstance(dataset_path, Path)
        assert dataset_path.exists()


    def test_custom_num_objects(self, tmp_path):
        """Test configurable range of objects per image."""
        min_objects = 3
        max_objects = 7
        dataset_path = BaseDataModule.create_synthetic_dataset(
            tmp_path, num_samples=20, split_ratio=0.7, min_objects=min_objects, max_objects=max_objects
        )

        # Check that each label file has objects within the specified range
        label_files = list((dataset_path / "labels" / "train").glob("*.txt"))

        for label_file in label_files:
            with open(label_file) as f:
                lines = [ln.strip() for ln in f.readlines()]
                # Filter out empty lines
                lines = [ln for ln in lines if ln]
                # Each image should have objects within the specified range
                assert min_objects <= len(lines) <= max_objects


class TestOBBDataset:
    """Tests for OBBDataset class."""

    def test_load_standard_detection_format(self, bounding_box_dataset):
        """Test loading standard detection format (5 values: class x y w h)."""
        # Load dataset
        dataset = OBBDataset(bounding_box_dataset, "train", img_size=640, num_classes=2)

        # Get first item
        img_tensor, labels = dataset[0]

        # Verify image shape
        assert img_tensor.shape == (3, 640, 640)
        assert img_tensor.dtype == torch.float32

        # Verify labels shape and content
        assert labels.shape == (2, 6)  # 2 boxes, 6 values each (cls, x, y, w, h, angle)

        # Check that rotation is 0 for standard detection format
        assert labels[0, 5].item() == 0.0  # First box rotation
        assert labels[1, 5].item() == 0.0  # Second box rotation

        # Check class labels
        assert labels[0, 0].item() == 0.0
        assert labels[1, 0].item() == 1.0

    def test_load_obb_format(self, oriented_bounding_box_dataset):
        """Test loading OBB format (9 values: class + 8 corner coordinates)."""
        # Load dataset
        dataset = OBBDataset(oriented_bounding_box_dataset, "train", img_size=640, num_classes=2)

        # Get first item
        img_tensor, labels = dataset[0]

        # Verify labels shape
        assert labels.shape == (1, 6)  # 1 box, 6 values

        # Check class label
        assert labels[0, 0].item() == 0.0

        # For axis-aligned rectangle, rotation should be close to 0
        assert abs(labels[0, 5].item()) < 0.1

    def test_standard_detection_warning_raised(self, yolo_dataset_dir, create_test_image):
        """Test that warning is raised when standard detection format is detected."""
        # Create test image
        img_path = yolo_dataset_dir / "images" / "train" / "test3.jpg"
        create_test_image(img_path)

        # Create label file with standard detection format
        label_path = yolo_dataset_dir / "labels" / "train" / "test3.txt"
        with open(label_path, "w", encoding="utf_8") as f:
            f.write("0 0.5 0.5 0.3 0.4\n")

        # Check that warning is raised
        with pytest.warns(UserWarning, match="Standard detection format detected"):
            dataset = OBBDataset(yolo_dataset_dir, "train", img_size=640, num_classes=2)
            # Load first item to trigger warning
            _, _ = dataset[0]

    def test_standard_detection_warning_only_once(self, yolo_dataset_dir, create_test_image):
        """Test that warning is only raised once per dataset instance."""
        # Create multiple test images with standard detection format
        for i in range(3):
            img_path = yolo_dataset_dir / "images" / "train" / f"test{i}.jpg"
            create_test_image(img_path)

            label_path = yolo_dataset_dir / "labels" / "train" / f"test{i}.txt"
            with open(label_path, "w", encoding="utf_8") as f:
                f.write(f"{i % 2} 0.5 0.5 0.3 0.4\n")

        # Check that warning is raised exactly once
        with pytest.warns(UserWarning, match="Standard detection format detected") as warning_list:
            dataset = OBBDataset(yolo_dataset_dir, "train", img_size=640, num_classes=2)

            # Load all items
            for i in range(len(dataset)):
                _, _ = dataset[i]

        # Warning should be raised exactly once
        assert len(warning_list) == 1

    def test_mixed_format_handling(self, mixed_detection_dataset):
        """Test that dataset can handle files without both formats mixed in same label file."""
        # Load dataset
        dataset = OBBDataset(mixed_detection_dataset, "train", img_size=640, num_classes=2)

        # Both images should load successfully
        assert len(dataset) == 2

        # Load both items
        _, labels1 = dataset[0]
        _, labels2 = dataset[1]

        # Both should have valid labels
        assert labels1.shape[0] > 0
        assert labels2.shape[0] > 0

    def test_invalid_format_handling(self, yolo_dataset_dir, create_test_image):
        """Test that invalid label lines are skipped with debug warnings."""
        # Create test image
        img_path = yolo_dataset_dir / "images" / "train" / "test_invalid.jpg"
        create_test_image(img_path)

        # Create label file with mixed valid and invalid lines
        label_path = yolo_dataset_dir / "labels" / "train" / "test_invalid.txt"
        with open(label_path, "w", encoding="utf_8") as f:
            f.write("0 0.5 0.5 0.3 0.4\n")  # Valid standard detection
            f.write("0 0.1 0.2 0.3\n")  # Invalid - unsupported format (4 values)
            f.write("abc 0.5 0.5 0.3 0.4\n")  # Invalid - non-numeric class
            f.write("0 0.3 0.3 0.7 0.3 0.7 0.7 0.3 0.7\n")  # Valid OBB
            f.write("1 a b c d e f g h\n")  # Invalid - non-numeric coords

        # Check that warnings are raised for invalid formats
        with pytest.warns(UserWarning):
            dataset = OBBDataset(yolo_dataset_dir, "train", img_size=640, num_classes=2)
            _, labels = dataset[0]

        # Should have 2 valid labels (1 standard + 1 OBB)
        assert labels.shape[0] == 2

    def test_empty_label_file(self, yolo_dataset_dir, create_test_image):
        """Test handling of empty label files."""
        # Create test image
        img_path = yolo_dataset_dir / "images" / "train" / "test_empty.jpg"
        create_test_image(img_path)

        # Create empty label file
        label_path = yolo_dataset_dir / "labels" / "train" / "test_empty.txt"
        label_path.touch()

        # Load dataset
        dataset = OBBDataset(yolo_dataset_dir, "train", img_size=640, num_classes=2)

        # Get item - should return empty labels
        _, labels = dataset[0]

        # Should have zero labels
        assert labels.shape == (0, 6)

    def test_missing_label_file(self, yolo_dataset_dir, create_test_image):
        """Test handling of missing label files."""
        # Create test image without corresponding label
        img_path = yolo_dataset_dir / "images" / "train" / "test_no_label.jpg"
        create_test_image(img_path)

        # Load dataset
        dataset = OBBDataset(yolo_dataset_dir, "train", img_size=640, num_classes=2)

        # Get item - should return empty labels
        _, labels = dataset[0]

        # Should have zero labels
        assert labels.shape == (0, 6)
