"""
YOLO-OBB Training with PyTorch Lightning

Features: logging, jsonargparse CLI, TorchMetrics mAP (train+val), AMP, auto class detection.

Usage:
    python -m lit_yolo --data /path/to/dataset --model yolo11n-obb.pt
    python -m lit_yolo --config config.yaml

Dependencies:
    pip install pytorch-lightning ultralytics "jsonargparse[signatures]" "torchmetrics[detection]"
"""

from __future__ import annotations

import logging

# Configure logging for the package
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Import main components
from .data import OBBDataModule, YOLOOBBDataset, corners_to_xywhr, detect_num_classes, obb_to_xyxy
from .models import YOLOOBBLightning
from .training import train

__all__ = [
    "OBBDataModule",
    "YOLOOBBDataset",
    "YOLOOBBLightning",
    "corners_to_xywhr",
    "detect_num_classes",
    "obb_to_xyxy",
    "train",
]

__version__ = "0.1.0"
