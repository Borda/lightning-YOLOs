"""Unit tests for visualization functionality."""


import cv2
import numpy as np
import pytest

from lit_yolo.data import DetDataModule, OBBDataModule
from lit_yolo.data.utils import create_batch_grid, draw_bboxes_on_image, draw_obb_on_image


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


class TestCreateBatchGrid:
    """Tests for create_batch_grid function."""

    def test_basic_grid(self):
        """Test basic grid creation."""
        imgs = [np.ones((100, 100, 3), dtype=np.uint8) * i for i in range(4)]
        grid = create_batch_grid(imgs, grid_size=(2, 2))

        # Grid should be larger than individual images due to borders
        assert grid.shape[0] > 100
        assert grid.shape[1] > 100

    def test_auto_grid_size(self):
        """Test automatic grid size determination."""
        imgs = [np.ones((100, 100, 3), dtype=np.uint8) for _ in range(6)]
        grid = create_batch_grid(imgs)

        # Should create grid (roughly square)
        assert grid.shape[0] > 0
        assert grid.shape[1] > 0

    def test_single_image(self):
        """Test grid with single image."""
        imgs = [np.ones((100, 100, 3), dtype=np.uint8) * 128]
        grid = create_batch_grid(imgs)

        assert grid.shape[0] > 100  # Includes border
        assert grid.shape[1] > 100

    def test_empty_images_list(self):
        """Test that empty list raises error."""
        with pytest.raises(ValueError, match="empty"):
            create_batch_grid([])


class TestOBBDataModuleVisualization:
    """Tests for OBBDataModule visualize_batch method."""

    @pytest.mark.parametrize("split,batch_size", [("train", 4), ("val", 2)])
    def test_visualize_batch(self, tmp_path, split, batch_size):
        """Test visualizing training and validation batches."""
        root = tmp_path / "synthetic"
        _ = OBBDataModule.create_synthetic_dataset(root, num_samples=10, split_ratio=0.7)

        dm = OBBDataModule(data=str(root), img_size=320, batch_size=batch_size, num_classes=3)
        dm.setup("fit")

        grid = dm.visualize_batch(split)

        # Check grid is created
        assert grid.shape[0] > 0
        assert grid.shape[1] > 0
        assert grid.shape[2] == 3

    def test_save_visualization(self, tmp_path):
        """Test saving visualization to file."""
        root = tmp_path / "synthetic"
        output_path = tmp_path / "output" / "viz.jpg"
        _ = OBBDataModule.create_synthetic_dataset(root, num_samples=10)

        dm = OBBDataModule(data=str(root), img_size=320, batch_size=4, num_classes=3)
        dm.setup("fit")

        grid = dm.visualize_batch("train", output_path=output_path)

        # Check file is created
        assert output_path.exists()
        # Check can read the saved image
        saved_img = cv2.imread(str(output_path))
        assert saved_img is not None
        assert saved_img.shape == grid.shape

    def test_with_class_names(self, tmp_path):
        """Test visualization with class names."""
        root = tmp_path / "synthetic"
        _ = OBBDataModule.create_synthetic_dataset(root, num_samples=10, class_mode="shape")

        dm = OBBDataModule(data=str(root), img_size=320, batch_size=4, num_classes=3)
        dm.setup("fit")

        class_names = ["square", "triangle", "circle"]
        grid = dm.visualize_batch("train", class_names=class_names)

        assert grid.shape[0] > 0
        assert grid.shape[1] > 0

    def test_invalid_batch_idx(self, tmp_path):
        """Test that invalid batch index raises error."""
        root = tmp_path / "synthetic"
        _ = OBBDataModule.create_synthetic_dataset(root, num_samples=4)

        dm = OBBDataModule(data=str(root), img_size=320, batch_size=4, num_classes=3)
        dm.setup("fit")

        # Only 1 batch with 4 samples and batch_size=4
        with pytest.raises(ValueError, match="out of range"):
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

        grid = dm.visualize_batch(split)

        assert grid.shape[0] > 0
        assert grid.shape[1] > 0
        assert grid.shape[2] == 3

    def test_save_visualization(self, tmp_path):
        """Test saving visualization to file."""
        root = tmp_path / "synthetic"
        output_path = tmp_path / "output" / "viz.jpg"
        _ = DetDataModule.create_synthetic_dataset(root, num_samples=10)

        dm = DetDataModule(data=str(root), img_size=320, batch_size=4, num_classes=3)
        dm.setup("fit")

        grid = dm.visualize_batch("train", output_path=output_path)

        # Check file is created
        assert output_path.exists()
        # Check can read the saved image
        saved_img = cv2.imread(str(output_path))
        assert saved_img is not None
        assert saved_img.shape == grid.shape

    def test_with_class_names(self, tmp_path):
        """Test visualization with class names."""
        root = tmp_path / "synthetic"
        _ = DetDataModule.create_synthetic_dataset(root, num_samples=10, class_mode="color")

        dm = DetDataModule(data=str(root), img_size=320, batch_size=4, num_classes=3)
        dm.setup("fit")

        class_names = ["red", "green", "blue"]
        grid = dm.visualize_batch("train", class_names=class_names)

        assert grid.shape[0] > 0
        assert grid.shape[1] > 0
