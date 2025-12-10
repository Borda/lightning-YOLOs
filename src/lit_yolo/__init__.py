"""
YOLO-OBB and standard detection training with PyTorch Lightning

Features: logging, jsonargparse CLI, TorchMetrics mAP (train+val), AMP, auto class detection.
"""

# Import main components
from lit_yolo.data import (
    DetDataModule,
    OBBDataModule,
    YOLODetDataset,
    YOLOOBBDataset,
    corners_to_xywhr,
    determine_num_classes,
    obb_to_xyxy,
    xywh_to_xyxy,
)
from lit_yolo.models import LitYOLODet, LitYOLOOBB
from lit_yolo.training import train_detect, train_obb

__all__ = [
    "DetDataModule",
    "OBBDataModule",
    "YOLODetDataset",
    "YOLOOBBDataset",
    "LitYOLODet",
    "LitYOLOOBB",
    "corners_to_xywhr",
    "determine_num_classes",
    "obb_to_xyxy",
    "xywh_to_xyxy",
    "train_obb",
    "train_detect",
]

__version__ = "0.1.0"
