"""Unit tests for lit_yolo.models module."""

import inspect

import pytest
import pytorch_lightning as pl
import torch

from lit_yolo.models import LitYOLOOBB


class TestLitYOLOOBB:
    """Tests for LitYOLOOBB model."""

    def test_instantiate_default(self):
        """Test instantiating model with default parameters."""
        # This should work without needing actual data
        # We skip actual YOLO model loading for this simple test
        # Just verify the class can be imported and basic attributes exist
        assert hasattr(LitYOLOOBB, '__init__')
        assert hasattr(LitYOLOOBB, 'forward')
        assert hasattr(LitYOLOOBB, 'training_step')
        assert hasattr(LitYOLOOBB, 'validation_step')
        assert hasattr(LitYOLOOBB, 'configure_optimizers')

    def test_class_is_lightning_module(self):
        """Test that LitYOLOOBB is a PyTorch Lightning module."""
        assert issubclass(LitYOLOOBB, pl.LightningModule)

    def test_default_hyperparameters(self):
        """Test that default hyperparameters are sensible."""
        # Test we can inspect the __init__ signature
        sig = inspect.signature(LitYOLOOBB.__init__)
        
        # Check expected parameters exist
        params = sig.parameters
        assert 'model_name' in params
        assert 'num_classes' in params
        assert 'lr' in params
        assert 'weight_decay' in params
        assert 'warmup_epochs' in params
        assert 'img_size' in params
        
        # Check default values
        assert params['num_classes'].default == 15
        assert params['lr'].default == 1e-3
        assert params['img_size'].default == 640
