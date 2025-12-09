"""
YOLO-OBB Training with PyTorch Lightning

Features: logging, jsonargparse CLI, TorchMetrics mAP (train+val), AMP, auto class detection.
"""

# Import main components
from lit_yolo.data import OBBDataModule, YOLOOBBDataset, corners_to_xywhr, detect_num_classes, obb_to_xyxy
from lit_yolo.models import LitYOLOOBB
from lit_yolo.training import train

__all__ = [
    "OBBDataModule",
    "YOLOOBBDataset",
    "LitYOLOOBB",
    "corners_to_xywhr",
    "detect_num_classes",
    "obb_to_xyxy",
    "train",
]

__version__ = "0.1.0"
