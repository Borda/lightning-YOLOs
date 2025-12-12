"""Unit tests for overlap threshold functionality in synthetic dataset generation."""

import numpy as np
import pytest

from lit_yolo.data.utils import (
    calculate_bbox_iou,
    calculate_boundary_overlap,
    generate_synthetic_sample,
)


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
