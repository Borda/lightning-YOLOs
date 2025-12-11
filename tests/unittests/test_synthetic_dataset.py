"""Unit tests for synthetic dataset creation."""

from pathlib import Path

import cv2
import pytest

from lit_yolo.data import BaseYOLODataModule, DetDataModule, OBBDataModule


class TestCreateSyntheticDataset:
    """Tests for create_synthetic_dataset static method."""

    def test_basic_creation(self, tmp_path):
        """Test basic synthetic dataset creation."""
        root = tmp_path / "synthetic"
        dataset_path = BaseYOLODataModule.create_synthetic_dataset(
            root, num_train=5, num_val=2, img_size=640, class_mode="shape"
        )

        # Verify path is returned correctly
        assert dataset_path == root

        # Verify directory structure
        assert (dataset_path / "images" / "train").exists()
        assert (dataset_path / "images" / "val").exists()
        assert (dataset_path / "labels" / "train").exists()
        assert (dataset_path / "labels" / "val").exists()

        # Verify number of images
        train_images = list((dataset_path / "images" / "train").glob("*.jpg"))
        val_images = list((dataset_path / "images" / "val").glob("*.jpg"))
        assert len(train_images) == 5
        assert len(val_images) == 2

        # Verify number of label files
        train_labels = list((dataset_path / "labels" / "train").glob("*.txt"))
        val_labels = list((dataset_path / "labels" / "val").glob("*.txt"))
        assert len(train_labels) == 5
        assert len(val_labels) == 2

    def test_shape_class_mode(self, tmp_path):
        """Test synthetic dataset with shape-based classes."""
        root = tmp_path / "synthetic_shape"
        dataset_path = BaseYOLODataModule.create_synthetic_dataset(
            root, num_train=10, num_val=5, class_mode="shape"
        )

        # Check that labels contain only class indices 0, 1, 2 (for 3 shapes)
        label_files = list((dataset_path / "labels" / "train").glob("*.txt"))
        classes_found = set()

        for label_file in label_files:
            with open(label_file) as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        parts = line.split()
                        cls = int(parts[0])
                        classes_found.add(cls)

        # Should have 3 classes (0, 1, 2) for shapes
        assert classes_found == {0, 1, 2}

    def test_color_class_mode(self, tmp_path):
        """Test synthetic dataset with color-based classes."""
        root = tmp_path / "synthetic_color"
        dataset_path = BaseYOLODataModule.create_synthetic_dataset(
            root, num_train=10, num_val=5, class_mode="color"
        )

        # Check that labels contain only class indices 0, 1, 2 (for 3 colors)
        label_files = list((dataset_path / "labels" / "train").glob("*.txt"))
        classes_found = set()

        for label_file in label_files:
            with open(label_file) as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        parts = line.split()
                        cls = int(parts[0])
                        classes_found.add(cls)

        # Should have 3 classes (0, 1, 2) for colors
        assert classes_found == {0, 1, 2}

    def test_invalid_class_mode(self, tmp_path):
        """Test that invalid class_mode raises ValueError."""
        root = tmp_path / "synthetic_invalid"
        with pytest.raises(ValueError, match="class_mode must be 'shape' or 'color'"):
            BaseYOLODataModule.create_synthetic_dataset(root, class_mode="invalid")

    def test_label_format_validity(self, tmp_path):
        """Test that generated labels are in valid YOLO format."""
        root = tmp_path / "synthetic_format"
        dataset_path = BaseYOLODataModule.create_synthetic_dataset(root, num_train=5, num_val=2)

        # Check label format (class cx cy w h, all normalized)
        label_files = list((dataset_path / "labels" / "train").glob("*.txt"))

        for label_file in label_files:
            with open(label_file) as f:
                lines = [line.strip() for line in f if line.strip()]
                # Each image should have 3 objects
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
        root = tmp_path / "synthetic_img"
        img_size = 512
        dataset_path = BaseYOLODataModule.create_synthetic_dataset(
            root, num_train=3, num_val=1, img_size=img_size
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
        root = tmp_path / "synthetic_corr"
        dataset_path = BaseYOLODataModule.create_synthetic_dataset(root, num_train=10, num_val=5)

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
        root1 = tmp_path / "synthetic_seed1"
        root2 = tmp_path / "synthetic_seed2"

        # Create two datasets with same seed
        BaseYOLODataModule.create_synthetic_dataset(root1, num_train=5, num_val=2, seed=123)
        BaseYOLODataModule.create_synthetic_dataset(root2, num_train=5, num_val=2, seed=123)

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
        root = tmp_path / "synthetic_custom_size"
        custom_size = 320
        dataset_path = BaseYOLODataModule.create_synthetic_dataset(
            root, num_train=3, num_val=1, img_size=custom_size
        )

        # Check that images have the custom size
        img_files = list((dataset_path / "images" / "train").glob("*.jpg"))
        for img_file in img_files:
            img = cv2.imread(str(img_file))
            assert img.shape[0] == custom_size
            assert img.shape[1] == custom_size

    def test_can_load_with_det_datamodule(self, tmp_path):
        """Test that synthetic dataset can be loaded with DetDataModule."""
        root = tmp_path / "synthetic_det"
        dataset_path = BaseYOLODataModule.create_synthetic_dataset(root, num_train=10, num_val=5)

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
        root = tmp_path / "synthetic_obb"
        dataset_path = BaseYOLODataModule.create_synthetic_dataset(root, num_train=10, num_val=5)

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
        root_str = str(tmp_path / "synthetic_str")
        dataset_path = BaseYOLODataModule.create_synthetic_dataset(root_str, num_train=3, num_val=1)

        # Should return Path object
        assert isinstance(dataset_path, Path)
        assert dataset_path.exists()

    def test_objects_per_image(self, tmp_path):
        """Test that each image contains exactly 3 objects (one of each shape)."""
        root = tmp_path / "synthetic_objects"
        dataset_path = BaseYOLODataModule.create_synthetic_dataset(root, num_train=5, num_val=2)

        # Check that each label file has 3 objects
        label_files = list((dataset_path / "labels" / "train").glob("*.txt"))

        for label_file in label_files:
            with open(label_file) as f:
                lines = [line.strip() for line in f if line.strip()]
                # Each image should have exactly 3 objects
                assert len(lines) == 3
