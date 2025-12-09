"""
Data utilities, dataset, and data module for YOLO-OBB.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl

logger = logging.getLogger(__name__)


# =============================================================================
# UTILITIES
# =============================================================================


def detect_num_classes(root: Path) -> int:
    """Scan label files to find max class index + 1."""
    max_class, files_scanned = -1, 0
    for split in ["train", "val"]:
        label_dir = root / "labels" / split
        if not label_dir.exists():
            continue
        for lf in label_dir.glob("*.txt"):
            try:
                with open(lf) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            max_class = max(max_class, int(parts[0]))
                files_scanned += 1
            except (ValueError, IOError):
                continue

    if max_class < 0:
        raise ValueError(f"No valid labels found in {root}/labels/")
    logger.info(f"Detected {max_class + 1} classes from {files_scanned} label files")
    return max_class + 1


def corners_to_xywhr(corners: np.ndarray) -> tuple[float, float, float, float, float]:
    """Convert 4 corner points to (cx, cy, w, h, angle) format."""
    SCALE = 1000.0
    corners_px = (corners * SCALE).astype(np.float32)
    (cx_px, cy_px), (w_px, h_px), angle_deg = cv2.minAreaRect(corners_px)

    cx, cy = cx_px / SCALE, cy_px / SCALE
    w, h = w_px / SCALE, h_px / SCALE

    if w < h:
        w, h = h, w
        angle_deg += 90

    # Clamp angle to [0, pi/2) with a small epsilon to avoid edge cases where angle == pi/2,
    # which can cause issues in downstream calculations (e.g., with trigonometric functions).
    angle_rad = max(0.0, min((angle_deg % 90) * math.pi / 180.0, math.pi / 2 - 1e-6))
    return cx, cy, w, h, angle_rad


def obb_to_xyxy(obb: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Convert OBB (xywhr) to axis-aligned xyxy bounding box."""
    if len(obb) == 0:
        return torch.empty((0, 4), device=obb.device)

    cx, cy, w, h, angle = obb[:, 0] * scale, obb[:, 1] * scale, obb[:, 2] * scale, obb[:, 3] * scale, obb[:, 4]
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    hw, hh = w / 2, h / 2

    dx = torch.stack([hw, hw, -hw, -hw], dim=1)
    dy = torch.stack([hh, -hh, -hh, hh], dim=1)
    corners_x = cx[:, None] + dx * cos_a[:, None] - dy * sin_a[:, None]
    corners_y = cy[:, None] + dx * sin_a[:, None] + dy * cos_a[:, None]

    return torch.stack([corners_x.min(1).values, corners_y.min(1).values,
                        corners_x.max(1).values, corners_y.max(1).values], dim=1)


# =============================================================================
# DATASET
# =============================================================================


class YOLOOBBDataset(Dataset):
    """Dataset for YOLO OBB format (4-corner annotations)."""

    FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, root: Path, split: str, img_size: int, num_classes: int):
        if not isinstance(img_size, int) or img_size <= 0:
            raise ValueError(f"img_size must be a positive integer, got {img_size}")
        self.img_size, self.num_classes = img_size, num_classes
        self.img_dir = root / "images" / split
        self.label_dir = root / "labels" / split

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        self.img_paths = sorted(p for p in self.img_dir.iterdir() if p.suffix.lower() in self.FORMATS)
        if not self.img_paths:
            raise ValueError(f"No images found in {self.img_dir}")
        logger.info(f"[{split}] Loaded {len(self.img_paths)} images")

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.img_paths[idx]

        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        labels = self._load_labels(self.label_dir / f"{img_path.stem}.txt")
        img, scale, (pad_w, pad_h) = self._letterbox(img)

        if labels.numel() > 0:
            new_w, new_h = orig_w * scale, orig_h * scale
            labels[:, 1] = (labels[:, 1] * new_w + pad_w) / self.img_size
            labels[:, 2] = (labels[:, 2] * new_h + pad_h) / self.img_size
            labels[:, 3] *= new_w / self.img_size
            labels[:, 4] *= new_h / self.img_size

        return torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0), labels

    def _load_labels(self, path: Path) -> torch.Tensor:
        if not path.exists():
            return torch.zeros((0, 6), dtype=torch.float32)

        labels = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 9:
                    continue
                try:
                    cls = int(parts[0])
                    if not (0 <= cls < self.num_classes):
                        continue
                    corners = np.array([float(x) for x in parts[1:9]], dtype=np.float32).reshape(4, 2)
                    labels.append([cls, *corners_to_xywhr(corners)])
                except ValueError:
                    continue

        return torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 6), dtype=torch.float32)

    def _letterbox(self, img: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        h, w = img.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w, pad_h = (self.img_size - new_w) // 2, (self.img_size - new_h) // 2
        img_padded = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        img_padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = img_resized
        return img_padded, scale, (pad_w, pad_h)


# =============================================================================
# DATA MODULE
# =============================================================================


class OBBDataModule(pl.LightningDataModule):
    """Lightning DataModule for OBB datasets - handles all data setup."""

    def __init__(self, data: str, img_size: int = 640, batch_size: int = 8, num_workers: int = 4, num_classes: int | None = None):
        super().__init__()
        self.data_root = Path(data)
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._num_classes = num_classes

    @property
    def num_classes(self) -> int:
        if self._num_classes is None:
            self._num_classes = detect_num_classes(self.data_root)
        return self._num_classes

    def setup(self, stage: str | None = None):
        nc = self.num_classes  # Triggers detection if needed
        self.train_ds = YOLOOBBDataset(self.data_root, "train", self.img_size, nc)
        self.val_ds = YOLOOBBDataset(self.data_root, "val", self.img_size, nc)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
            collate_fn=self._collate, persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=self._collate, persistent_workers=self.num_workers > 0
        )

    @staticmethod
    def _collate(batch: list[tuple]) -> dict[str, Any]:
        imgs, batch_idx, cls_list, bbox_list = [], [], [], []

        for i, (img, labels) in enumerate(batch):
            imgs.append(img)
            if labels.numel() > 0:
                n = labels.shape[0]
                batch_idx.append(torch.full((n,), i, dtype=torch.long))
                cls_list.append(labels[:, 0:1])
                bbox_list.append(labels[:, 1:6])

        return {
            "img": torch.stack(imgs),
            "batch_idx": torch.cat(batch_idx) if batch_idx else torch.empty(0, dtype=torch.long),
            "cls": torch.cat(cls_list) if cls_list else torch.empty(0, 1),
            "bboxes": torch.cat(bbox_list) if bbox_list else torch.empty(0, 5),
        }
