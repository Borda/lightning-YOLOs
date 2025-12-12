"""Unit tests for visualization functionality."""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytest

from lit_yolo.data import DetDataModule, OBBDataModule, draw_bboxes_on_image, draw_obb_on_image


class TestDrawBboxesOnImage:
    """Tests for draw_bboxes_on_image function."""

    def test_basic_drawing(self):
        """Test basic bounding box drawing."""
        img = np.ones((640, 640, 3), dtype=np.uint8) * 128
        bboxes = np.array([[0.5, 0.5, 0.2, 0.3], [0.3, 0.3, 0.1, 0.1]])
        class_ids = np.array([0, 1])

        result = draw_bboxes_on_image(img, bboxes, class_ids)

        # Check shape is preserved
        assert result.shape == (640, 640, 3)
        # Check image is modified (not same as input)
        assert not np.array_equal(result, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def test_with_class_names(self):
        """Test drawing with class names."""
        img = np.ones((640, 640, 3), dtype=np.uint8) * 128
        bboxes = np.array([[0.5, 0.5, 0.2, 0.3]])
        class_ids = np.array([0])
        class_names = ["cat", "dog"]

        result = draw_bboxes_on_image(img, bboxes, class_ids, class_names=class_names)

        assert result.shape == (640, 640, 3)

    def test_empty_boxes(self):
        """Test with no bounding boxes."""
        img = np.ones((640, 640, 3), dtype=np.uint8) * 128
        bboxes = np.array([]).reshape(0, 4)
        class_ids = np.array([])

        result = draw_bboxes_on_image(img, bboxes, class_ids)

        # Should return image converted to BGR
        assert result.shape == (640, 640, 3)

    def test_invalid_image_shape(self):
        """Test that invalid image shape raises error."""
        img = np.ones((640, 640), dtype=np.uint8)  # 2D image
        bboxes = np.array([[0.5, 0.5, 0.2, 0.3]])
        class_ids = np.array([0])

        with pytest.raises(ValueError, match="Image must have shape"):
            draw_bboxes_on_image(img, bboxes, class_ids)


class TestDrawOBBOnImage:
    """Tests for draw_obb_on_image function."""

    def test_basic_drawing(self):
        """Test basic OBB drawing."""
        img = np.ones((640, 640, 3), dtype=np.uint8) * 128
        obbs = np.array([[0.5, 0.5, 0.2, 0.3, 0.0], [0.3, 0.3, 0.1, 0.1, np.pi / 4]])
        class_ids = np.array([0, 1])

        result = draw_obb_on_image(img, obbs, class_ids)

        # Check shape is preserved
        assert result.shape == (640, 640, 3)
        # Check image is modified
        assert not np.array_equal(result, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def test_with_class_names(self):
        """Test OBB drawing with class names."""
        img = np.ones((640, 640, 3), dtype=np.uint8) * 128
        obbs = np.array([[0.5, 0.5, 0.2, 0.3, 0.0]])
        class_ids = np.array([0])
        class_names = ["car", "truck"]

        result = draw_obb_on_image(img, obbs, class_ids, class_names=class_names)

        assert result.shape == (640, 640, 3)

    def test_rotated_box(self):
        """Test drawing rotated bounding box."""
        img = np.ones((640, 640, 3), dtype=np.uint8) * 128
        # Box rotated by 45 degrees
        obbs = np.array([[0.5, 0.5, 0.2, 0.3, np.pi / 4]])
        class_ids = np.array([0])

        result = draw_obb_on_image(img, obbs, class_ids)

        assert result.shape == (640, 640, 3)

    def test_empty_boxes(self):
        """Test with no oriented bounding boxes."""
        img = np.ones((640, 640, 3), dtype=np.uint8) * 128
        obbs = np.array([]).reshape(0, 5)
        class_ids = np.array([])

        result = draw_obb_on_image(img, obbs, class_ids)

        assert result.shape == (640, 640, 3)


class TestOBBDataModuleVisualization:
    """Tests for OBBDataModule visualize_batch method."""

    @pytest.mark.parametrize("split,batch_size", [("train", 4), ("val", 2)])
    def test_visualize_batch(self, tmp_path, split, batch_size):
        """Test visualizing training and validation batches."""
        root = tmp_path / "synthetic"
        _ = OBBDataModule.create_synthetic_dataset(root, num_samples=10, split_ratio=0.7)

        dm = OBBDataModule(data=str(root), img_size=320, batch_size=batch_size, num_classes=3)
        dm.setup("fit")

        fig, axes = dm.visualize_batch(split)

        # Check figure and axes are created
        assert fig is not None
        assert axes is not None
        # Clean up
        fig.clf()
        plt.close(fig)

    def test_save_visualization(self, tmp_path):
        """Test saving visualization to file."""
        root = tmp_path / "synthetic"
        output_path = tmp_path / "output" / "viz.jpg"
        _ = OBBDataModule.create_synthetic_dataset(root, num_samples=10)

        dm = OBBDataModule(data=str(root), img_size=320, batch_size=4, num_classes=3)
        dm.setup("fit")

        fig, axes = dm.visualize_batch("train")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), bbox_inches="tight")

        fig.clf()
        plt.close(fig)

        # Check file is created
        assert output_path.exists()
        # Check can read the saved image
        saved_img = cv2.imread(str(output_path))
        assert saved_img is not None
        assert saved_img.shape[2] == 3  # Check that the saved image has 3 channels

    def test_with_class_names(self, tmp_path):
        """Test visualization with class names."""
        root = tmp_path / "synthetic"
        _ = OBBDataModule.create_synthetic_dataset(root, num_samples=10, class_mode="shape")

        dm = OBBDataModule(data=str(root), img_size=320, batch_size=4, num_classes=3)
        dm.setup("fit")

        class_names = ["square", "triangle", "circle"]
        dm._class_names = class_names
        fig, axes = dm.visualize_batch("train")

        assert fig is not None
        assert axes is not None
        # Clean up
        fig.clf()
        plt.close(fig)

    def test_invalid_batch_idx(self, tmp_path):
        """Test that invalid batch index raises error."""
        root = tmp_path / "synthetic"
        _ = OBBDataModule.create_synthetic_dataset(root, num_samples=4)

        dm = OBBDataModule(data=str(root), img_size=320, batch_size=4, num_classes=3)
        dm.setup("fit")

        # Only 1 batch with 4 samples and batch_size=4
        with pytest.raises(IndexError, match="out of range"):
            dm.visualize_batch("train", batch_idx=10)


class TestDetDataModuleVisualization:
    """Tests for DetDataModule visualize_batch method."""

    @pytest.mark.parametrize("split,batch_size", [("train", 4), ("val", 2)])
    def test_visualize_batch(self, tmp_path, split, batch_size):
        """Test visualizing training and validation batches."""
        root = tmp_path / "synthetic"
        _ = DetDataModule.create_synthetic_dataset(root, num_samples=10, split_ratio=0.7)

        dm = DetDataModule(data=str(root), img_size=320, batch_size=batch_size, num_classes=3)
        dm.setup("fit")

        fig, axes = dm.visualize_batch(split)

        assert fig is not None
        assert axes is not None
        # Clean up
        fig.clf()
        plt.close(fig)

    def test_save_visualization(self, tmp_path):
        """Test saving visualization to file."""
        root = tmp_path / "synthetic"
        output_path = tmp_path / "output" / "viz.jpg"
        _ = DetDataModule.create_synthetic_dataset(root, num_samples=10)

        dm = DetDataModule(data=str(root), img_size=320, batch_size=4, num_classes=3)
        dm.setup("fit")

        fig, axes = dm.visualize_batch("train")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), bbox_inches="tight")

        fig.clf()
        plt.close(fig)

        # Check file is created
        assert output_path.exists()
        # Check can read the saved image
        saved_img = cv2.imread(str(output_path))
        assert saved_img is not None
        assert saved_img.shape[2] == 3

    def test_with_class_names(self, tmp_path):
        """Test visualization with class names."""
        root = tmp_path / "synthetic"
        _ = DetDataModule.create_synthetic_dataset(root, num_samples=10, class_mode="color")

        dm = DetDataModule(data=str(root), img_size=320, batch_size=4, num_classes=3)
        dm.setup("fit")

        class_names = ["red", "green", "blue"]
        dm._class_names = class_names
        fig, axes = dm.visualize_batch("train")

        assert fig is not None
        assert axes is not None
        # Clean up
        fig.clf()
        plt.close(fig)
