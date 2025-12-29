"""Unit tests for lit_yolo.models module."""

from unittest.mock import patch

import pytest
import torch

from lit_yolo.models import LitYOLODet, LitYOLOOBB

# Constants for test data dimensions
# YOLO model output format: (batch_size, num_channels, num_detections)
# OBB format: num_channels = num_classes + 5 (x, y, w, h, angle)
# Detection format: num_channels = num_classes + 4 (x, y, w, h)
OBB_EXTRA_CHANNELS = 5  # xywhr
DET_EXTRA_CHANNELS = 4  # xywh
NUM_DETECTIONS = 100  # Typical number of detection candidates


class TestLitYOLOOBBProcessMethods:
    """Tests for LitYOLOOBB box processing methods."""

    @pytest.fixture
    def obb_model(self):
        """Create a LitYOLOOBB model for testing."""
        return LitYOLOOBB(model_name="yolo11n-obb.pt", num_classes=10, img_size=640)

    def test_is_rotated_property(self, obb_model):
        """Test that is_rotated returns True for OBB model."""
        assert obb_model.is_rotated is True

    def test_process_target_boxes_empty(self, obb_model):
        """Test _process_target_boxes handles empty input correctly."""
        gt_box = torch.empty((0, 5))
        result = obb_model._process_target_boxes(gt_box)

        assert result.shape == (0, 4)
        assert result.device == obb_model.device

    def test_process_target_boxes_single_box(self, obb_model):
        """Test _process_target_boxes converts single OBB to xyxy format."""
        # Normalized xywhr: center at (0.5, 0.5), size (0.2, 0.1), angle 0
        gt_box = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.0]])
        result = obb_model._process_target_boxes(gt_box)

        # Should be converted to pixel coordinates and xyxy format
        assert result.shape == (1, 4)
        # Check that coordinates are in valid range (0 to img_size)
        assert (result >= 0).all()
        assert (result[:, [0, 2]] <= obb_model.img_size).all()
        assert (result[:, [1, 3]] <= obb_model.img_size).all()
        # Check that x1 < x2 and y1 < y2
        assert (result[:, 0] < result[:, 2]).all()
        assert (result[:, 1] < result[:, 3]).all()

    def test_process_target_boxes_multiple_boxes(self, obb_model):
        """Test _process_target_boxes handles multiple boxes correctly."""
        # Multiple normalized xywhr boxes
        gt_box = torch.tensor(
            [
                [0.5, 0.5, 0.2, 0.1, 0.0],  # Center box, no rotation
                [0.25, 0.25, 0.1, 0.1, 0.0],  # Top-left box, no rotation
                [0.75, 0.75, 0.15, 0.15, 0.785],  # Bottom-right rotated box (π/4 radians ≈ 45 degrees)
            ]
        )
        result = obb_model._process_target_boxes(gt_box)

        assert result.shape == (3, 4)
        # All coordinates should be valid
        assert (result >= 0).all()
        # Check that x1 < x2 and y1 < y2 for all boxes
        assert (result[:, 0] < result[:, 2]).all()
        assert (result[:, 1] < result[:, 3]).all()

    def test_process_pred_boxes_empty(self, obb_model):
        """Test _process_pred_boxes handles empty predictions correctly."""
        pred = torch.empty((0, 7))  # Empty predictions with 7 columns (xywhr, conf, cls)
        boxes, scores, labels = obb_model._process_pred_boxes(pred)

        assert boxes.shape == (0, 4)
        assert scores.shape == (0,)
        assert labels.shape == (0,)
        assert labels.dtype == torch.long

    def test_process_pred_boxes_single_prediction(self, obb_model):
        """Test _process_pred_boxes processes single OBB prediction
        correctly."""
        # Prediction in pixel coordinates: xywhr (center, size, angle), conf, cls
        pred = torch.tensor([[320.0, 320.0, 100.0, 50.0, 0.0, 0.95, 2.0]])
        boxes, scores, labels = obb_model._process_pred_boxes(pred)

        assert boxes.shape == (1, 4)
        assert scores.shape == (1,)
        assert labels.shape == (1,)
        assert labels.dtype == torch.long

        # Check score and label values
        assert scores[0] == 0.95
        assert labels[0] == 2

        # Check boxes are clamped to image bounds
        assert (boxes >= 0).all()
        assert (boxes[:, [0, 2]] <= obb_model.img_size).all()
        assert (boxes[:, [1, 3]] <= obb_model.img_size).all()

    def test_process_pred_boxes_multiple_predictions(self, obb_model):
        """Test _process_pred_boxes handles multiple predictions correctly."""
        # Multiple predictions
        pred = torch.tensor(
            [
                [320.0, 320.0, 100.0, 50.0, 0.0, 0.95, 2.0],  # No rotation
                [100.0, 100.0, 80.0, 60.0, 0.0, 0.85, 0.0],  # No rotation
                [500.0, 500.0, 120.0, 80.0, 0.785, 0.75, 5.0],  # Rotated π/4 radians (~45 degrees)
            ]
        )
        boxes, scores, labels = obb_model._process_pred_boxes(pred)

        assert boxes.shape == (3, 4)
        assert scores.shape == (3,)
        assert labels.shape == (3,)

        # Check all scores and labels
        assert torch.allclose(scores, torch.tensor([0.95, 0.85, 0.75]))
        assert torch.equal(labels, torch.tensor([2, 0, 5], dtype=torch.long))

    def test_process_pred_boxes_clamping(self, obb_model):
        """Test that _process_pred_boxes clamps boxes to image bounds."""
        # Prediction that would exceed image bounds
        pred = torch.tensor([[50.0, 50.0, 200.0, 200.0, 0.0, 0.90, 1.0]])
        boxes, scores, labels = obb_model._process_pred_boxes(pred)

        # Boxes should be clamped to [0, img_size]
        assert (boxes[:, [0, 2]] >= 0).all()
        assert (boxes[:, [0, 2]] <= obb_model.img_size).all()
        assert (boxes[:, [1, 3]] >= 0).all()
        assert (boxes[:, [1, 3]] <= obb_model.img_size).all()


class TestLitYOLODetProcessMethods:
    """Tests for LitYOLODet box processing methods."""

    @pytest.fixture
    def det_model(self):
        """Create a LitYOLODet model for testing."""
        return LitYOLODet(model_name="yolo11n.pt", num_classes=80, img_size=640)

    def test_is_rotated_property(self, det_model):
        """Test that is_rotated returns False for detection model."""
        assert det_model.is_rotated is False

    def test_process_target_boxes_empty(self, det_model):
        """Test _process_target_boxes handles empty input correctly."""
        gt_box = torch.empty((0, 4))
        result = det_model._process_target_boxes(gt_box)

        assert result.shape == (0, 4)
        assert result.device == det_model.device

    def test_process_target_boxes_single_box(self, det_model):
        """Test _process_target_boxes converts single box to xyxy format."""
        # Normalized xywh: center at (0.5, 0.5), size (0.2, 0.1)
        gt_box = torch.tensor([[0.5, 0.5, 0.2, 0.1]])
        result = det_model._process_target_boxes(gt_box)

        # Should be converted to pixel coordinates and xyxy format
        assert result.shape == (1, 4)
        # Check that coordinates are in valid range
        assert (result >= 0).all()
        assert (result[:, [0, 2]] <= det_model.img_size).all()
        assert (result[:, [1, 3]] <= det_model.img_size).all()
        # Check that x1 < x2 and y1 < y2
        assert (result[:, 0] < result[:, 2]).all()
        assert (result[:, 1] < result[:, 3]).all()

    def test_process_target_boxes_multiple_boxes(self, det_model):
        """Test _process_target_boxes handles multiple boxes correctly."""
        # Multiple normalized xywh boxes
        gt_box = torch.tensor(
            [
                [0.5, 0.5, 0.2, 0.1],  # Center box
                [0.25, 0.25, 0.1, 0.1],  # Top-left box
                [0.75, 0.75, 0.15, 0.15],  # Bottom-right box
            ]
        )
        result = det_model._process_target_boxes(gt_box)

        assert result.shape == (3, 4)
        # All coordinates should be valid
        assert (result >= 0).all()
        # Check that x1 < x2 and y1 < y2 for all boxes
        assert (result[:, 0] < result[:, 2]).all()
        assert (result[:, 1] < result[:, 3]).all()

    def test_process_pred_boxes_empty(self, det_model):
        """Test _process_pred_boxes handles empty predictions correctly."""
        pred = torch.empty((0, 6))  # Empty predictions with 6 columns (xyxy, conf, cls)
        boxes, scores, labels = det_model._process_pred_boxes(pred)

        assert boxes.shape == (0, 4)
        assert scores.shape == (0,)
        assert labels.shape == (0,)
        assert labels.dtype == torch.long

    def test_process_pred_boxes_single_prediction(self, det_model):
        """Test _process_pred_boxes processes single detection correctly."""
        # Prediction in pixel coordinates: xyxy, conf, cls
        pred = torch.tensor([[100.0, 100.0, 300.0, 250.0, 0.95, 2.0]])
        boxes, scores, labels = det_model._process_pred_boxes(pred)

        assert boxes.shape == (1, 4)
        assert scores.shape == (1,)
        assert labels.shape == (1,)
        assert labels.dtype == torch.long

        # Check score and label values
        assert scores[0] == 0.95
        assert labels[0] == 2

        # Check boxes are clamped to image bounds
        assert (boxes >= 0).all()
        assert (boxes[:, [0, 2]] <= det_model.img_size).all()
        assert (boxes[:, [1, 3]] <= det_model.img_size).all()

    def test_process_pred_boxes_multiple_predictions(self, det_model):
        """Test _process_pred_boxes handles multiple predictions correctly."""
        # Multiple predictions
        pred = torch.tensor(
            [
                [100.0, 100.0, 300.0, 250.0, 0.95, 2.0],
                [50.0, 50.0, 150.0, 120.0, 0.85, 0.0],
                [400.0, 400.0, 550.0, 520.0, 0.75, 5.0],
            ]
        )
        boxes, scores, labels = det_model._process_pred_boxes(pred)

        assert boxes.shape == (3, 4)
        assert scores.shape == (3,)
        assert labels.shape == (3,)

        # Check all scores and labels
        assert torch.allclose(scores, torch.tensor([0.95, 0.85, 0.75]))
        assert torch.equal(labels, torch.tensor([2, 0, 5], dtype=torch.long))

    def test_process_pred_boxes_clamping(self, det_model):
        """Test that _process_pred_boxes clamps boxes to image bounds."""
        # Prediction that exceeds image bounds
        pred = torch.tensor([[-10.0, -10.0, 700.0, 700.0, 0.90, 1.0]])
        boxes, scores, labels = det_model._process_pred_boxes(pred)

        # Boxes should be clamped to [0, img_size]
        assert (boxes[:, [0, 2]] >= 0).all()
        assert (boxes[:, [0, 2]] <= det_model.img_size).all()
        assert (boxes[:, [1, 3]] >= 0).all()
        assert (boxes[:, [1, 3]] <= det_model.img_size).all()


class TestUpdateMetricsLogic:
    """Tests for _update_metrics method behavior.

    These tests verify that the base class metric update logic correctly
    handles batch processing and formatting for torchmetrics, including
    edge cases with empty predictions and targets.
    """

    @pytest.fixture
    def obb_model(self):
        """Create a LitYOLOOBB model for testing."""
        model = LitYOLOOBB(model_name="yolo11n-obb.pt", num_classes=10, img_size=640)
        # Call setup to initialize metrics
        model.setup(stage="fit")
        return model

    @pytest.fixture
    def det_model(self):
        """Create a LitYOLODet model for testing."""
        model = LitYOLODet(model_name="yolo11n.pt", num_classes=80, img_size=640)
        # Call setup to initialize metrics
        model.setup(stage="fit")
        return model

    @patch("lit_yolo.models.non_max_suppression")
    def test_update_metrics_with_empty_predictions_obb(self, mock_nms, obb_model):
        """Test _update_metrics handles batches where NMS returns empty predictions."""
        # Mock NMS to return empty predictions for all images in batch
        mock_nms.return_value = [
            torch.empty((0, 7)),  # Image 0: no predictions
            torch.empty((0, 7)),  # Image 1: no predictions
        ]

        # Create batch with targets
        batch = {
            "batch_idx": torch.tensor([0, 0, 1]),
            "bboxes": torch.tensor(
                [[0.5, 0.5, 0.2, 0.1, 0.0], [0.3, 0.3, 0.1, 0.1, 0.0], [0.7, 0.7, 0.15, 0.15, 0.0]]
            ),
            "cls": torch.tensor([[0], [1], [2]]),
        }

        # Create mock predictions (raw model output)
        # Format: (batch_size, num_classes + extra_channels, num_detections)
        preds = torch.zeros((2, obb_model.nc + OBB_EXTRA_CHANNELS, NUM_DETECTIONS))

        # This should not raise an error
        obb_model._update_metrics(preds, batch, obb_model.train_map)

        # Verify NMS was called with correct parameters
        mock_nms.assert_called_once()
        call_kwargs = mock_nms.call_args[1]
        assert call_kwargs["rotated"] is True
        assert call_kwargs["nc"] == 10

    @patch("lit_yolo.models.non_max_suppression")
    def test_update_metrics_with_empty_targets_obb(self, mock_nms, obb_model):
        """Test _update_metrics handles batches with no ground truth targets."""
        # Mock NMS to return predictions
        mock_nms.return_value = [
            torch.tensor([[320.0, 320.0, 100.0, 50.0, 0.0, 0.95, 2.0]]),  # Image 0: one prediction
            torch.tensor([[100.0, 100.0, 80.0, 60.0, 0.0, 0.85, 0.0]]),  # Image 1: one prediction
        ]

        # Create batch with NO targets
        batch = {
            "batch_idx": torch.empty(0, dtype=torch.long),
            "bboxes": torch.empty((0, 5)),
            "cls": torch.empty((0, 1)),
        }

        # Create mock predictions
        # Format: (batch_size, num_classes + extra_channels, num_detections)
        preds = torch.zeros((2, obb_model.nc + OBB_EXTRA_CHANNELS, NUM_DETECTIONS))

        # This should not raise an error
        obb_model._update_metrics(preds, batch, obb_model.train_map)

        # Verify NMS was called
        mock_nms.assert_called_once()

    @patch("lit_yolo.models.non_max_suppression")
    def test_update_metrics_with_mixed_batch_obb(self, mock_nms, obb_model):
        """Test _update_metrics handles batches where some images have no targets."""
        # Mock NMS to return predictions for both images
        mock_nms.return_value = [
            torch.tensor([[320.0, 320.0, 100.0, 50.0, 0.0, 0.95, 2.0]]),  # Image 0: one prediction
            torch.tensor([[100.0, 100.0, 80.0, 60.0, 0.0, 0.85, 0.0]]),  # Image 1: one prediction
        ]

        # Create batch where only image 1 has targets (image 0 is empty)
        batch = {
            "batch_idx": torch.tensor([1, 1]),  # Only image 1 has targets
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.0], [0.3, 0.3, 0.1, 0.1, 0.0]]),
            "cls": torch.tensor([[0], [1]]),
        }

        # Create mock predictions
        # Format: (batch_size, num_classes + extra_channels, num_detections)
        preds = torch.zeros((2, obb_model.nc + OBB_EXTRA_CHANNELS, NUM_DETECTIONS))

        # This should not raise an error
        obb_model._update_metrics(preds, batch, obb_model.train_map)

        # Verify NMS was called
        mock_nms.assert_called_once()

    @patch("lit_yolo.models.non_max_suppression")
    def test_update_metrics_with_empty_predictions_det(self, mock_nms, det_model):
        """Test _update_metrics handles batches where NMS returns empty predictions."""
        # Mock NMS to return empty predictions
        mock_nms.return_value = [
            torch.empty((0, 6)),  # Image 0: no predictions
            torch.empty((0, 6)),  # Image 1: no predictions
        ]

        # Create batch with targets
        batch = {
            "batch_idx": torch.tensor([0, 0, 1]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.1], [0.3, 0.3, 0.1, 0.1], [0.7, 0.7, 0.15, 0.15]]),
            "cls": torch.tensor([[0], [1], [2]]),
        }

        # Create mock predictions
        # Format: (batch_size, num_classes + extra_channels, num_detections)
        preds = torch.zeros((2, det_model.nc + DET_EXTRA_CHANNELS, NUM_DETECTIONS))

        # This should not raise an error
        det_model._update_metrics(preds, batch, det_model.train_map)

        # Verify NMS was called with correct parameters
        mock_nms.assert_called_once()
        call_kwargs = mock_nms.call_args[1]
        assert call_kwargs["rotated"] is False
        assert call_kwargs["nc"] == 80

    @patch("lit_yolo.models.non_max_suppression")
    def test_update_metrics_with_empty_targets_det(self, mock_nms, det_model):
        """Test _update_metrics handles batches with no ground truth targets."""
        # Mock NMS to return predictions
        mock_nms.return_value = [
            torch.tensor([[100.0, 100.0, 300.0, 250.0, 0.95, 2.0]]),  # Image 0: one prediction
            torch.tensor([[50.0, 50.0, 150.0, 120.0, 0.85, 0.0]]),  # Image 1: one prediction
        ]

        # Create batch with NO targets
        batch = {
            "batch_idx": torch.empty(0, dtype=torch.long),
            "bboxes": torch.empty((0, 4)),
            "cls": torch.empty((0, 1)),
        }

        # Create mock predictions
        # Format: (batch_size, num_classes + extra_channels, num_detections)
        preds = torch.zeros((2, det_model.nc + DET_EXTRA_CHANNELS, NUM_DETECTIONS))

        # This should not raise an error
        det_model._update_metrics(preds, batch, det_model.train_map)

        # Verify NMS was called
        mock_nms.assert_called_once()

    @patch("lit_yolo.models.non_max_suppression")
    def test_update_metrics_with_mixed_batch_det(self, mock_nms, det_model):
        """Test _update_metrics handles batches where some images have no targets."""
        # Mock NMS to return predictions for both images
        mock_nms.return_value = [
            torch.tensor([[100.0, 100.0, 300.0, 250.0, 0.95, 2.0]]),  # Image 0: one prediction
            torch.tensor([[50.0, 50.0, 150.0, 120.0, 0.85, 0.0]]),  # Image 1: one prediction
        ]

        # Create batch where only image 1 has targets (image 0 is empty)
        batch = {
            "batch_idx": torch.tensor([1, 1]),  # Only image 1 has targets
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.1], [0.3, 0.3, 0.1, 0.1]]),
            "cls": torch.tensor([[0], [1]]),
        }

        # Create mock predictions
        # Format: (batch_size, num_classes + extra_channels, num_detections)
        preds = torch.zeros((2, det_model.nc + DET_EXTRA_CHANNELS, NUM_DETECTIONS))

        # This should not raise an error
        det_model._update_metrics(preds, batch, det_model.train_map)

        # Verify NMS was called
        mock_nms.assert_called_once()
