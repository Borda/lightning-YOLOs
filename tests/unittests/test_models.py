"""Unit tests for lit_yolo.models module."""

import pytest
import torch

from lit_yolo.models import BaseLitYOLO, LitYOLODet, LitYOLOOBB


class TestBaseLitYOLOAbstractMethods:
    """Tests for abstract method enforcement in BaseLitYOLO."""

    def test_is_rotated_not_implemented(self):
        """Test that is_rotated property raises NotImplementedError when not overridden."""

        class IncompleteYOLO(BaseLitYOLO):
            """Minimal implementation without overriding abstract methods."""

            pass

        model = IncompleteYOLO(model_name="yolo11n.pt", num_classes=10)
        with pytest.raises(NotImplementedError):
            _ = model.is_rotated

    def test_process_target_boxes_not_implemented(self):
        """Test that _process_target_boxes raises NotImplementedError when not overridden."""

        class IncompleteYOLO(BaseLitYOLO):
            """Minimal implementation without overriding abstract methods."""

            @property
            def is_rotated(self) -> bool:
                return False

        model = IncompleteYOLO(model_name="yolo11n.pt", num_classes=10)
        gt_box = torch.zeros((1, 4))
        with pytest.raises(NotImplementedError):
            model._process_target_boxes(gt_box)

    def test_process_pred_boxes_not_implemented(self):
        """Test that _process_pred_boxes raises NotImplementedError when not overridden."""

        class IncompleteYOLO(BaseLitYOLO):
            """Minimal implementation without overriding abstract methods."""

            @property
            def is_rotated(self) -> bool:
                return False

        model = IncompleteYOLO(model_name="yolo11n.pt", num_classes=10)
        pred = torch.zeros((1, 6))
        with pytest.raises(NotImplementedError):
            model._process_pred_boxes(pred)


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
                [0.5, 0.5, 0.2, 0.1, 0.0],  # Center box
                [0.25, 0.25, 0.1, 0.1, 0.0],  # Top-left box
                [0.75, 0.75, 0.15, 0.15, 0.785],  # Bottom-right rotated box (45 degrees)
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
        """Test _process_pred_boxes processes single OBB prediction correctly."""
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
                [320.0, 320.0, 100.0, 50.0, 0.0, 0.95, 2.0],
                [100.0, 100.0, 80.0, 60.0, 0.0, 0.85, 0.0],
                [500.0, 500.0, 120.0, 80.0, 0.785, 0.75, 5.0],
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

    These tests verify that the base class metric update logic correctly handles
    batch processing and formatting for torchmetrics.
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

    def test_process_empty_batch_obb(self, obb_model):
        """Test that OBB model handles empty batch indices correctly."""
        # Test that empty ground truth boxes are handled correctly
        gt_box_empty = torch.empty((0, 5))
        result = obb_model._process_target_boxes(gt_box_empty)
        assert result.shape == (0, 4)

        # Simulate batch processing with empty targets for an image
        batch = {
            "batch_idx": torch.tensor([1, 1]),  # Only image 1 has targets, image 0 is empty
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.0], [0.3, 0.3, 0.1, 0.1, 0.0]]),
            "cls": torch.tensor([[0], [1]]),
        }

        # For image 0, mask will be empty
        mask = batch["batch_idx"] == 0
        gt_box = batch["bboxes"][mask]
        assert len(gt_box) == 0
        result = obb_model._process_target_boxes(gt_box)
        assert result.shape == (0, 4)

    def test_process_empty_batch_det(self, det_model):
        """Test that detection model handles empty batch indices correctly."""
        # Test that empty ground truth boxes are handled correctly
        gt_box_empty = torch.empty((0, 4))
        result = det_model._process_target_boxes(gt_box_empty)
        assert result.shape == (0, 4)

        # Simulate batch processing with empty targets for an image
        batch = {
            "batch_idx": torch.tensor([1, 1]),  # Only image 1 has targets, image 0 is empty
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.1], [0.3, 0.3, 0.1, 0.1]]),
            "cls": torch.tensor([[0], [1]]),
        }

        # For image 0, mask will be empty
        mask = batch["batch_idx"] == 0
        gt_box = batch["bboxes"][mask]
        assert len(gt_box) == 0
        result = det_model._process_target_boxes(gt_box)
        assert result.shape == (0, 4)

    def test_empty_predictions_create_correct_format_obb(self, obb_model):
        """Test that empty predictions are formatted correctly for OBB."""
        # When NMS returns None or empty predictions, we should create empty tensors
        empty_pred = torch.empty((0, 7))
        boxes, scores, labels = obb_model._process_pred_boxes(empty_pred)

        assert boxes.shape == (0, 4)
        assert scores.shape == (0,)
        assert labels.shape == (0,)
        assert labels.dtype == torch.long

    def test_empty_predictions_create_correct_format_det(self, det_model):
        """Test that empty predictions are formatted correctly for detection."""
        # When NMS returns None or empty predictions, we should create empty tensors
        empty_pred = torch.empty((0, 6))
        boxes, scores, labels = det_model._process_pred_boxes(empty_pred)

        assert boxes.shape == (0, 4)
        assert scores.shape == (0,)
        assert labels.shape == (0,)
        assert labels.dtype == torch.long
