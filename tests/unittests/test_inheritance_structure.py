"""Tests to verify the inheritance structure and method availability."""

import inspect

import pytest


class TestDatasetInheritance:
    """Test Dataset inheritance structure."""

    def test_obb_dataset_has_all_required_methods(self):
        """Test that YOLOOBBDataset has all required methods after refactoring."""
        from lit_yolo.data import YOLOOBBDataset

        # Check that all essential methods are present
        assert hasattr(YOLOOBBDataset, "__init__")
        assert hasattr(YOLOOBBDataset, "__len__")
        assert hasattr(YOLOOBBDataset, "__getitem__")
        assert hasattr(YOLOOBBDataset, "_load_labels")
        assert hasattr(YOLOOBBDataset, "_letterbox")

        # Verify _load_labels is not abstract in the concrete class
        assert not inspect.isabstract(YOLOOBBDataset)

    def test_det_dataset_has_all_required_methods(self):
        """Test that YOLODetDataset has all required methods after refactoring."""
        from lit_yolo.data import YOLODetDataset

        # Check that all essential methods are present
        assert hasattr(YOLODetDataset, "__init__")
        assert hasattr(YOLODetDataset, "__len__")
        assert hasattr(YOLODetDataset, "__getitem__")
        assert hasattr(YOLODetDataset, "_load_labels")
        assert hasattr(YOLODetDataset, "_letterbox")

        # Verify _load_labels is not abstract in the concrete class
        assert not inspect.isabstract(YOLODetDataset)

    def test_base_dataset_methods_are_shared(self):
        """Test that BaseYOLODataset provides shared methods."""
        from lit_yolo.data import BaseYOLODataset, YOLODetDataset, YOLOOBBDataset

        # _letterbox should be the same method in both subclasses (inherited from base)
        assert YOLOOBBDataset._letterbox is BaseYOLODataset._letterbox
        assert YOLODetDataset._letterbox is BaseYOLODataset._letterbox


class TestDataModuleInheritance:
    """Test DataModule inheritance structure."""

    def test_obb_datamodule_has_all_required_methods(self):
        """Test that OBBDataModule has all required methods after refactoring."""
        from lit_yolo.data import OBBDataModule

        assert hasattr(OBBDataModule, "__init__")
        assert hasattr(OBBDataModule, "num_classes")
        assert hasattr(OBBDataModule, "setup")
        assert hasattr(OBBDataModule, "train_dataloader")
        assert hasattr(OBBDataModule, "val_dataloader")
        assert hasattr(OBBDataModule, "_collate")

    def test_det_datamodule_has_all_required_methods(self):
        """Test that DetDataModule has all required methods after refactoring."""
        from lit_yolo.data import DetDataModule

        assert hasattr(DetDataModule, "__init__")
        assert hasattr(DetDataModule, "num_classes")
        assert hasattr(DetDataModule, "setup")
        assert hasattr(DetDataModule, "train_dataloader")
        assert hasattr(DetDataModule, "val_dataloader")
        assert hasattr(DetDataModule, "_collate")

    def test_base_datamodule_methods_are_shared(self):
        """Test that BaseYOLODataModule provides shared methods."""
        from lit_yolo.data import BaseYOLODataModule, DetDataModule, OBBDataModule

        # train_dataloader should be the same method in both subclasses (inherited from base)
        assert OBBDataModule.train_dataloader is BaseYOLODataModule.train_dataloader
        assert DetDataModule.train_dataloader is BaseYOLODataModule.train_dataloader


class TestModelInheritance:
    """Test Model inheritance structure."""

    def test_obb_model_has_all_required_methods(self):
        """Test that LitYOLOOBB has all required methods after refactoring."""
        from lit_yolo.models import LitYOLOOBB

        assert hasattr(LitYOLOOBB, "__init__")
        assert hasattr(LitYOLOOBB, "setup")
        assert hasattr(LitYOLOOBB, "forward")
        assert hasattr(LitYOLOOBB, "training_step")
        assert hasattr(LitYOLOOBB, "validation_step")
        assert hasattr(LitYOLOOBB, "configure_optimizers")
        assert hasattr(LitYOLOOBB, "_compute_step")
        assert hasattr(LitYOLOOBB, "_update_metrics")
        assert hasattr(LitYOLOOBB, "_log_metrics")

    def test_det_model_has_all_required_methods(self):
        """Test that LitYOLODet has all required methods after refactoring."""
        from lit_yolo.models import LitYOLODet

        assert hasattr(LitYOLODet, "__init__")
        assert hasattr(LitYOLODet, "setup")
        assert hasattr(LitYOLODet, "forward")
        assert hasattr(LitYOLODet, "training_step")
        assert hasattr(LitYOLODet, "validation_step")
        assert hasattr(LitYOLODet, "configure_optimizers")
        assert hasattr(LitYOLODet, "_compute_step")
        assert hasattr(LitYOLODet, "_update_metrics")
        assert hasattr(LitYOLODet, "_log_metrics")

    def test_base_model_methods_are_shared(self):
        """Test that BaseLitYOLO provides shared methods."""
        from lit_yolo.models import BaseLitYOLO, LitYOLODet, LitYOLOOBB

        # Common methods should be the same in both subclasses (inherited from base)
        assert LitYOLOOBB.configure_optimizers is BaseLitYOLO.configure_optimizers
        assert LitYOLODet.configure_optimizers is BaseLitYOLO.configure_optimizers
        assert LitYOLOOBB.training_step is BaseLitYOLO.training_step
        assert LitYOLODet.training_step is BaseLitYOLO.training_step
        assert LitYOLOOBB._compute_step is BaseLitYOLO._compute_step
        assert LitYOLODet._compute_step is BaseLitYOLO._compute_step


class TestExportedBaseClasses:
    """Test that base classes are properly exported."""

    def test_base_classes_in_module_all(self):
        """Test that base classes are in __all__ export list."""
        import lit_yolo

        assert "BaseYOLODataset" in lit_yolo.__all__
        assert "BaseYOLODataModule" in lit_yolo.__all__
        assert "BaseLitYOLO" in lit_yolo.__all__

    def test_base_classes_can_be_imported(self):
        """Test that base classes can be imported from main module."""
        from lit_yolo import BaseLitYOLO, BaseYOLODataModule, BaseYOLODataset

        assert BaseLitYOLO is not None
        assert BaseYOLODataModule is not None
        assert BaseYOLODataset is not None
