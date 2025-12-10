"""Unit tests for lit_yolo.training module."""

import pytest

from lit_yolo.training import train


class TestTrain:
    """Tests for train function."""

    def test_train_function_exists(self):
        """Test that train function exists and is callable."""
        assert callable(train)
    def test_train_signature(self):
        """Test train function has expected parameters."""
        import inspect
        sig = inspect.signature(train)
        params = sig.parameters
        
        # Check core parameters exist
        assert 'data' in params
        assert 'model' in params
        assert 'epochs' in params
        assert 'batch_size' in params
        assert 'lr' in params
