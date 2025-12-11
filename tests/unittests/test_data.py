"""Unit tests for lit_yolo.data module."""

import math

import numpy as np
import pytest
import torch

from lit_yolo.data import YOLOOBBDataset, corners_to_xywhr, obb_to_xyxy, xywh_to_xyxy


class TestCornersToXywhr:
    """Tests for corners_to_xywhr function."""

    def test_square_box(self):
        """Test converting a square box from corners to xywhr format."""
        # Square box centered at (0.5, 0.5) with size 0.2x0.2
        corners = np.array(
            [
                [0.4, 0.4],
                [0.6, 0.4],
                [0.6, 0.6],
                [0.4, 0.6],
            ],
            dtype=np.float32,
        )

        cx, cy, w, h, angle = corners_to_xywhr(corners)

        # Check center is approximately correct
        assert abs(cx - 0.5) < 0.01
        assert abs(cy - 0.5) < 0.01
        # Check dimensions
        assert abs(w - 0.2) < 0.01
        assert abs(h - 0.2) < 0.01
        # Angle should be close to 0 for axis-aligned box
        assert abs(angle) < 0.1

    def test_rectangular_box(self):
        """Test converting a rectangular box."""
        # Rectangle 0.4 wide, 0.2 tall
        corners = np.array(
            [
                [0.3, 0.4],
                [0.7, 0.4],
                [0.7, 0.6],
                [0.3, 0.6],
            ],
            dtype=np.float32,
        )

        cx, cy, w, h, angle = corners_to_xywhr(corners)

        # Check center
        assert abs(cx - 0.5) < 0.01
        assert abs(cy - 0.5) < 0.01
        # Width should be larger dimension
        assert w > h
        assert abs(w - 0.4) < 0.01

    @pytest.mark.parametrize("angle_deg", [15, 30, 45, 60])
    def test_rotated_box(self, angle_deg):
        """Test converting a rotated box with different dimensions."""
        # Create a rectangle with width=0.3, height=0.1, rotated by angle_deg
        width, height = 0.3, 0.1
        angle_rad = math.radians(angle_deg)

        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Generate corners of rotated rectangle centered at (0.5, 0.5)
        hw, hh = width / 2, height / 2
        corners = np.array(
            [
                [0.5 + hw * cos_a - hh * sin_a, 0.5 + hw * sin_a + hh * cos_a],
                [0.5 + hw * cos_a + hh * sin_a, 0.5 + hw * sin_a - hh * cos_a],
                [0.5 - hw * cos_a + hh * sin_a, 0.5 - hw * sin_a - hh * cos_a],
                [0.5 - hw * cos_a - hh * sin_a, 0.5 - hw * sin_a + hh * cos_a],
            ],
            dtype=np.float32,
        )

        cx, cy, w, h, angle = corners_to_xywhr(corners)

        # Check center is approximately correct
        assert abs(cx - 0.5) < 0.01
        assert abs(cy - 0.5) < 0.01
        # Check dimensions (w should be larger)
        assert abs(w - width) < 0.01
        assert abs(h - height) < 0.01
        # Angle should be non-zero for rotated box
        assert 0 < angle < math.pi / 2


class TestObbToXyxy:
    """Tests for obb_to_xyxy function."""

    def test_empty_input(self):
        """Test with empty tensor."""
        obb = torch.empty((0, 5))
        result = obb_to_xyxy(obb)
        assert result.shape == (0, 4)

    def test_axis_aligned_box(self):
        """Test conversion of axis-aligned OBB to xyxy."""
        # Box centered at (0.5, 0.5), size 0.2x0.1, no rotation
        obb = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.0]])
        result = obb_to_xyxy(obb, scale=1.0)

        assert result.shape == (1, 4)
        # x1, y1 should be around (0.4, 0.45)
        # x2, y2 should be around (0.6, 0.55)
        assert abs(result[0, 0].item() - 0.4) < 0.01
        assert abs(result[0, 1].item() - 0.45) < 0.01
        assert abs(result[0, 2].item() - 0.6) < 0.01
        assert abs(result[0, 3].item() - 0.55) < 0.01

    def test_multiple_boxes(self):
        """Test conversion of multiple OBBs."""
        obb = torch.tensor(
            [
                [0.5, 0.5, 0.2, 0.1, 0.0],
                [0.3, 0.3, 0.1, 0.1, 0.0],
            ]
        )
        result = obb_to_xyxy(obb, scale=1.0)
        assert result.shape == (2, 4)

    def test_with_scaling(self):
        """Test conversion with scaling factor."""
        obb = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.0]])
        result = obb_to_xyxy(obb, scale=2.0)

        # All coordinates should be scaled by 2
        assert result.shape == (1, 4)
        assert abs(result[0, 0].item() - 0.8) < 0.02
        assert abs(result[0, 2].item() - 1.2) < 0.02

    @pytest.mark.parametrize("angle_deg", [15, 30, 45, 60])
    def test_rotated_box(self, angle_deg):
        """Test conversion of rotated OBB to xyxy."""
        # Box centered at (0.5, 0.5), size 0.2x0.1, rotated by angle_deg
        angle_rad = math.radians(angle_deg)
        obb = torch.tensor([[0.5, 0.5, 0.2, 0.1, angle_rad]])
        result = obb_to_xyxy(obb, scale=1.0)

        assert result.shape == (1, 4)
        # For a rotated box, the axis-aligned bounding box should be larger
        # than the original box dimensions
        width = result[0, 2].item() - result[0, 0].item()
        height = result[0, 3].item() - result[0, 1].item()

        # The bounding box should contain the rotated box
        assert width > 0.1  # Larger than original height
        assert height > 0.1  # Larger than original height
        assert width < 0.3  # But not unreasonably large
        assert height < 0.3


class TestXywhToXyxy:
    """Tests for xywh_to_xyxy function (standard detection)."""

    def test_empty_input(self):
        """Test with empty tensor."""
        bbox = torch.empty((0, 4))
        result = xywh_to_xyxy(bbox)
        assert result.shape == (0, 4)

    def test_single_box(self):
        """Test conversion of single axis-aligned box."""
        # Box centered at (0.5, 0.5), size 0.2x0.1
        bbox = torch.tensor([[0.5, 0.5, 0.2, 0.1]])
        result = xywh_to_xyxy(bbox, scale=1.0)

        assert result.shape == (1, 4)
        # x1, y1 should be around (0.4, 0.45)
        # x2, y2 should be around (0.6, 0.55)
        assert abs(result[0, 0].item() - 0.4) < 0.01
        assert abs(result[0, 1].item() - 0.45) < 0.01
        assert abs(result[0, 2].item() - 0.6) < 0.01
        assert abs(result[0, 3].item() - 0.55) < 0.01

    def test_square_box(self):
        """Test conversion of square box."""
        # Square box centered at (0.5, 0.5), size 0.2x0.2
        bbox = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        result = xywh_to_xyxy(bbox, scale=1.0)

        assert result.shape == (1, 4)
        assert abs(result[0, 0].item() - 0.4) < 0.01
        assert abs(result[0, 1].item() - 0.4) < 0.01
        assert abs(result[0, 2].item() - 0.6) < 0.01
        assert abs(result[0, 3].item() - 0.6) < 0.01

    def test_multiple_boxes(self):
        """Test conversion of multiple boxes."""
        bbox = torch.tensor(
            [
                [0.5, 0.5, 0.2, 0.1],
                [0.3, 0.3, 0.1, 0.1],
                [0.7, 0.7, 0.3, 0.2],
            ]
        )
        result = xywh_to_xyxy(bbox, scale=1.0)
        assert result.shape == (3, 4)

        # Verify first box
        assert abs(result[0, 0].item() - 0.4) < 0.01
        assert abs(result[0, 2].item() - 0.6) < 0.01

        # Verify second box
        assert abs(result[1, 0].item() - 0.25) < 0.01
        assert abs(result[1, 2].item() - 0.35) < 0.01

    def test_with_scaling(self):
        """Test conversion with scaling factor."""
        bbox = torch.tensor([[0.5, 0.5, 0.2, 0.1]])
        result = xywh_to_xyxy(bbox, scale=2.0)

        # All coordinates should be scaled by 2
        assert result.shape == (1, 4)
        assert abs(result[0, 0].item() - 0.8) < 0.02  # (0.5 - 0.2/2) * 2 = 0.8
        assert abs(result[0, 1].item() - 0.9) < 0.02  # (0.5 - 0.1/2) * 2 = 0.9
        assert abs(result[0, 2].item() - 1.2) < 0.02  # (0.5 + 0.2/2) * 2 = 1.2
        assert abs(result[0, 3].item() - 1.1) < 0.02  # (0.5 + 0.1/2) * 2 = 1.1

    def test_edge_cases(self):
        """Test edge cases with boxes at image boundaries."""
        # Box at top-left corner
        bbox = torch.tensor([[0.1, 0.1, 0.2, 0.2]])
        result = xywh_to_xyxy(bbox, scale=1.0)

        assert result.shape == (1, 4)
        assert abs(result[0, 0].item() - 0.0) < 0.01
        assert abs(result[0, 1].item() - 0.0) < 0.01
        assert abs(result[0, 2].item() - 0.2) < 0.01
        assert abs(result[0, 3].item() - 0.2) < 0.01

    def test_normalized_coordinates(self):
        """Test that function handles normalized coordinates properly."""
        # Typical YOLO normalized format (0-1 range)
        bbox = torch.tensor(
            [
                [0.25, 0.25, 0.5, 0.5],  # Quarter size box at quarter position
                [0.75, 0.75, 0.5, 0.5],  # Quarter size box at three-quarter position
            ]
        )
        result = xywh_to_xyxy(bbox, scale=1.0)

        assert result.shape == (2, 4)
        # First box should span from 0.0 to 0.5 in both dimensions
        assert abs(result[0, 0].item() - 0.0) < 0.01
        assert abs(result[0, 1].item() - 0.0) < 0.01
        assert abs(result[0, 2].item() - 0.5) < 0.01
        assert abs(result[0, 3].item() - 0.5) < 0.01

        # Second box should span from 0.5 to 1.0 in both dimensions
        assert abs(result[1, 0].item() - 0.5) < 0.01
        assert abs(result[1, 1].item() - 0.5) < 0.01
        assert abs(result[1, 2].item() - 1.0) < 0.01
        assert abs(result[1, 3].item() - 1.0) < 0.01

    def test_pixel_coordinates_with_scaling(self):
        """Test conversion with pixel coordinates (scale=640 for 640x640 image)."""
        # Normalized coordinates
        bbox = torch.tensor([[0.5, 0.5, 0.5, 0.25]])
        result = xywh_to_xyxy(bbox, scale=640.0)

        assert result.shape == (1, 4)
        # Expected: center (320, 320), size (320, 160)
        # x1 = 320 - 160 = 160, y1 = 320 - 80 = 240
        # x2 = 320 + 160 = 480, y2 = 320 + 80 = 400
        assert abs(result[0, 0].item() - 160) < 1
        assert abs(result[0, 1].item() - 240) < 1
        assert abs(result[0, 2].item() - 480) < 1
        assert abs(result[0, 3].item() - 400) < 1


class TestYOLOOBBDataset:
    """Tests for YOLOOBBDataset class."""

    def test_load_standard_detection_format(self, standard_detection_dataset):
        """Test loading standard detection format (5 values: class x y w h)."""
        # Load dataset
        dataset = YOLOOBBDataset(standard_detection_dataset, "train", img_size=640, num_classes=2)

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

    def test_load_obb_format(self, obb_format_dataset):
        """Test loading OBB format (9 values: class + 8 corner coordinates)."""
        # Load dataset
        dataset = YOLOOBBDataset(obb_format_dataset, "train", img_size=640, num_classes=2)

        # Get first item
        img_tensor, labels = dataset[0]

        # Verify labels shape
        assert labels.shape == (1, 6)  # 1 box, 6 values

        # Check class label
        assert labels[0, 0].item() == 0.0

        # For axis-aligned rectangle, rotation should be close to 0
        assert abs(labels[0, 5].item()) < 0.1

    def test_standard_detection_warning_raised(self, obb_dataset_dir, create_test_image):
        """Test that warning is raised when standard detection format is detected."""
        # Create test image
        img_path = obb_dataset_dir / "images" / "train" / "test3.jpg"
        create_test_image(img_path)

        # Create label file with standard detection format
        label_path = obb_dataset_dir / "labels" / "train" / "test3.txt"
        with open(label_path, "w") as f:
            f.write("0 0.5 0.5 0.3 0.4\n")

        # Check that warning is raised
        with pytest.warns(UserWarning, match="Standard detection format detected"):
            dataset = YOLOOBBDataset(obb_dataset_dir, "train", img_size=640, num_classes=2)
            # Load first item to trigger warning
            _, _ = dataset[0]

    def test_standard_detection_warning_only_once(self, obb_dataset_dir, create_test_image):
        """Test that warning is only raised once per dataset instance."""
        # Create multiple test images with standard detection format
        for i in range(3):
            img_path = obb_dataset_dir / "images" / "train" / f"test{i}.jpg"
            create_test_image(img_path)

            label_path = obb_dataset_dir / "labels" / "train" / f"test{i}.txt"
            with open(label_path, "w") as f:
                f.write(f"{i % 2} 0.5 0.5 0.3 0.4\n")

        # Check that warning is raised exactly once
        with pytest.warns(UserWarning, match="Standard detection format detected") as warning_list:
            dataset = YOLOOBBDataset(obb_dataset_dir, "train", img_size=640, num_classes=2)

            # Load all items
            for i in range(len(dataset)):
                _, _ = dataset[i]

        # Warning should be raised exactly once
        assert len(warning_list) == 1

    def test_mixed_format_handling(self, mixed_format_dataset):
        """Test that dataset can handle files without both formats mixed in same label file."""
        # Load dataset
        dataset = YOLOOBBDataset(mixed_format_dataset, "train", img_size=640, num_classes=2)

        # Both images should load successfully
        assert len(dataset) == 2

        # Load both items
        _, labels1 = dataset[0]
        _, labels2 = dataset[1]

        # Both should have valid labels
        assert labels1.shape[0] > 0
        assert labels2.shape[0] > 0

    def test_invalid_format_handling(self, obb_dataset_dir, create_test_image):
        """Test that invalid label lines are skipped with debug warnings."""
        # Create test image
        img_path = obb_dataset_dir / "images" / "train" / "test_invalid.jpg"
        create_test_image(img_path)

        # Create label file with mixed valid and invalid lines
        label_path = obb_dataset_dir / "labels" / "train" / "test_invalid.txt"
        with open(label_path, "w") as f:
            f.write("0 0.5 0.5 0.3 0.4\n")  # Valid standard detection
            f.write("0 0.1 0.2 0.3\n")  # Invalid - unsupported format (4 values)
            f.write("abc 0.5 0.5 0.3 0.4\n")  # Invalid - non-numeric class
            f.write("0 0.3 0.3 0.7 0.3 0.7 0.7 0.3 0.7\n")  # Valid OBB
            f.write("1 a b c d e f g h\n")  # Invalid - non-numeric coords

        # Check that warnings are raised for invalid formats
        with pytest.warns(UserWarning):
            dataset = YOLOOBBDataset(obb_dataset_dir, "train", img_size=640, num_classes=2)
            _, labels = dataset[0]

        # Should have 2 valid labels (1 standard + 1 OBB)
        assert labels.shape[0] == 2

    def test_empty_label_file(self, obb_dataset_dir, create_test_image):
        """Test handling of empty label files."""
        # Create test image
        img_path = obb_dataset_dir / "images" / "train" / "test_empty.jpg"
        create_test_image(img_path)

        # Create empty label file
        label_path = obb_dataset_dir / "labels" / "train" / "test_empty.txt"
        label_path.touch()

        # Load dataset
        dataset = YOLOOBBDataset(obb_dataset_dir, "train", img_size=640, num_classes=2)

        # Get item - should return empty labels
        _, labels = dataset[0]

        # Should have zero labels
        assert labels.shape == (0, 6)

    def test_missing_label_file(self, obb_dataset_dir, create_test_image):
        """Test handling of missing label files."""
        # Create test image without corresponding label
        img_path = obb_dataset_dir / "images" / "train" / "test_no_label.jpg"
        create_test_image(img_path)

        # Load dataset
        dataset = YOLOOBBDataset(obb_dataset_dir, "train", img_size=640, num_classes=2)

        # Get item - should return empty labels
        _, labels = dataset[0]

        # Should have zero labels
        assert labels.shape == (0, 6)
