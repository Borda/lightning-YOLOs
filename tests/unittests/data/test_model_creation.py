from lit_yolo.models import BaseLitYOLO
import torch
import pytest

@pytest.fixture
def model():
    return BaseLitYOLO("yolo11n-obb.pt", 10)

class TestModel:
    """Tests for creating model with Lightning"""

    def test_create_model(self, model):
        """Test creation of torch.nn.Module"""
        assert isinstance(model, torch.nn.Module)

    def test_output(self, model):
        """Check that output is a tuple and the tensors have correct size"""
        image = torch.rand([1, 3, 640, 640])
        output = model(image)
        assert isinstance(output, tuple)

        first_tuple = output[0]
        assert first_tuple[0].shape == (1, 74, 80, 80)
        assert first_tuple[1].shape == (1, 74, 40, 40)
        assert first_tuple[2].shape == (1, 74, 20, 20)

        second_tuple = output[1]
        assert second_tuple.shape == (1, 1, 8400)