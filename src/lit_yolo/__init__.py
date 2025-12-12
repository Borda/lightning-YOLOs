"""
YOLO-OBB and standard detection training with PyTorch Lightning

Features: logging, jsonargparse CLI, TorchMetrics mAP (train+val), AMP, auto class detection.
"""

# Import main components
from lit_yolo.data import (
    BaseDataModule,
    BaseDataset,
    DetDataModule,
    DetDataset,
    OBBDataModule,
    OBBDataset,
    corners_to_xywhr,
    determine_num_classes,
)
from lit_yolo.models import BaseLitYOLO, LitYOLODet, LitYOLOOBB
from lit_yolo.training import train_detect, train_obb

__all__ = [
    "BaseLitYOLO",
    "BaseDataModule",
    "BaseDataset",
    "DetDataModule",
    "DetDataset",
    "OBBDataModule",
    "OBBDataset",
    "LitYOLODet",
    "LitYOLOOBB",
    "corners_to_xywhr",
    "determine_num_classes",
    "train_obb",
    "train_detect",
]

__version__ = "0.1.0"
