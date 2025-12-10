"""Unit tests for lit_yolo.data module."""

import math

import numpy as np
import pytest
import torch

from lit_yolo.data import corners_to_xywhr, obb_to_xyxy


class TestCornersToXywhr:
    """Tests for corners_to_xywhr function."""

    def test_square_box(self):
        """Test converting a square box from corners to xywhr format."""
        # Square box centered at (0.5, 0.5) with size 0.2x0.2
        corners = np.array([
            [0.4, 0.4],
            [0.6, 0.4],
            [0.6, 0.6],
            [0.4, 0.6],
        ], dtype=np.float32)
        
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
        corners = np.array([
            [0.3, 0.4],
            [0.7, 0.4],
            [0.7, 0.6],
            [0.3, 0.6],
        ], dtype=np.float32)
        
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
        corners = np.array([
            [0.5 + hw * cos_a - hh * sin_a, 0.5 + hw * sin_a + hh * cos_a],
            [0.5 + hw * cos_a + hh * sin_a, 0.5 + hw * sin_a - hh * cos_a],
            [0.5 - hw * cos_a + hh * sin_a, 0.5 - hw * sin_a - hh * cos_a],
            [0.5 - hw * cos_a - hh * sin_a, 0.5 - hw * sin_a + hh * cos_a],
        ], dtype=np.float32)
        
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
        obb = torch.tensor([
            [0.5, 0.5, 0.2, 0.1, 0.0],
            [0.3, 0.3, 0.1, 0.1, 0.0],
        ])
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
