"""Unit tests for lit_yolo.data module."""

import numpy as np

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
