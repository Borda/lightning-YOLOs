"""
YOLO-OBB Training with PyTorch Lightning

Features: logging, jsonargparse CLI, TorchMetrics mAP (train+val), AMP, auto class detection.

Usage:
    python -m lit_yolo --data /path/to/dataset --model yolo11n-obb.pt
    python -m lit_yolo --config config.yaml

Dependencies:
    pip install pytorch-lightning ultralytics "jsonargparse[signatures]" "torchmetrics[detection]"
"""

# Import main components
from .data import OBBDataModule, YOLOOBBDataset, corners_to_xywhr, detect_num_classes, obb_to_xyxy
from .models import LitYOLOOBB
from .training import train

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
