"""
YOLO-OBB Training with PyTorch Lightning (v4)
==============================================

A production-ready implementation for training YOLO OBB (Oriented Bounding Box)
models using PyTorch Lightning with proper logging, CLI configuration, and metrics.

Features:
    - Logging with appropriate levels (DEBUG, INFO, WARNING, ERROR)
    - jsonargparse CLI with automatic argument parsing from function signatures
    - TorchMetrics MeanAveragePrecision for mAP@50, mAP@50:95 evaluation
    - Memory-efficient training with gradient clipping and AMP
    - Automatic class detection from dataset labels

Note on Metrics:
    OBB predictions are converted to axis-aligned bounding boxes for TorchMetrics
    compatibility. This provides a reasonable approximation for evaluation purposes.

Usage:
    python yolo_obb_lightning_v4.py --data /path/to/dataset --model yolo11n-obb.pt
    python yolo_obb_lightning_v4.py --config config.yaml
    python yolo_obb_lightning_v4.py --help

Dependencies:
    pip install pytorch-lightning ultralytics "jsonargparse[signatures]" "torchmetrics[detection]"
"""

from __future__ import annotations

import logging
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Module logger
logger = logging.getLogger(__name__)


# =============================================================================
# UTILITIES
# =============================================================================


def detect_num_classes(root: Union[str, Path]) -> int:
    """Scan label files to determine the number of classes.

    Iterates through train and val label directories to find the maximum
    class index, then returns max_index + 1.

    Args:
        root: Dataset root directory containing labels/train and labels/val.

    Returns:
        Number of classes (max class index + 1).

    Raises:
        ValueError: If no valid labels are found.
    """
    root = Path(root)
    max_class = -1
    files_scanned = 0

    for split in ["train", "val"]:
        label_dir = root / "labels" / split
        if not label_dir.exists():
            logger.debug(f"Label directory not found: {label_dir}")
            continue

        for label_file in label_dir.glob("*.txt"):
            try:
                with open(label_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            cls = int(parts[0])
                            max_class = max(max_class, cls)
                files_scanned += 1
            except (ValueError, IOError) as e:
                logger.debug(f"Error reading {label_file}: {e}")
                continue

    if max_class < 0:
        raise ValueError(f"No valid labels found in {root}/labels/")

    num_classes = max_class + 1
    logger.info(f"Detected {num_classes} classes from {files_scanned} label files")
    return num_classes


def corners_to_xywhr(corners: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Convert 4 corner points to xywhr format.

    Uses cv2.minAreaRect to compute the minimum area rotated rectangle
    from normalized corner coordinates.

    Args:
        corners: Array of shape [4, 2] with normalized corner coordinates.

    Returns:
        Tuple of (cx, cy, w, h, angle) where:
            - cx, cy: Normalized center coordinates
            - w, h: Normalized width and height (w >= h)
            - angle: Rotation angle in radians [0, π/2)
    """
    # Scale to virtual pixels for cv2.minAreaRect (expects pixel coords)
    SCALE = 1000.0
    corners_px = (corners * SCALE).astype(np.float32)

    rect = cv2.minAreaRect(corners_px)
    (cx_px, cy_px), (w_px, h_px), angle_deg = rect

    # Back to normalized coordinates
    cx, cy = cx_px / SCALE, cy_px / SCALE
    w, h = w_px / SCALE, h_px / SCALE

    # Ensure w >= h and normalize angle to [0, 90)
    if w < h:
        w, h = h, w
        angle_deg += 90

    angle_deg = angle_deg % 90
    if angle_deg < 0:
        angle_deg += 90

    # Convert to radians, clamp to [0, π/2)
    angle_rad = max(0.0, min(angle_deg * math.pi / 180.0, math.pi / 2 - 1e-6))

    return cx, cy, w, h, angle_rad


# =============================================================================
# DATASET
# =============================================================================


class YOLOOBBDataset(Dataset):
    """Dataset for YOLO OBB format with 4-corner annotations.

    Expected directory structure:
        root/
            images/
                train/
                val/
            labels/
                train/
                val/

    Label format (per line):
        class x1 y1 x2 y2 x3 y3 x4 y4

    Where coordinates are normalized [0, 1].

    Args:
        root: Dataset root directory.
        split: Data split ('train' or 'val').
        img_size: Target image size after letterbox resize.
        num_classes: Number of classes for filtering invalid labels.
    """

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        img_size: int = 640,
        num_classes: Optional[int] = None,
    ):
        self.root = Path(root)
        self.img_size = img_size
        self.num_classes = num_classes
        self.split = split

        self.img_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        self.img_paths = sorted(
            [p for p in self.img_dir.iterdir() if p.suffix.lower() in self.SUPPORTED_FORMATS]
        )

        if not self.img_paths:
            raise ValueError(f"No images found in {self.img_dir}")

        logger.info(f"[{split}] Loaded {len(self.img_paths)} images from {self.img_dir}")

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Load and preprocess image with labels.

        Returns:
            img_tensor: Float tensor [C, H, W] in range [0, 1].
            labels: Float tensor [N, 6] with [cls, cx, cy, w, h, angle].
            meta: Dict with 'img_path', 'orig_shape', 'scale', 'pad'.
        """
        img_path = self.img_paths[idx]

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        # Load labels
        label_path = self.label_dir / f"{img_path.stem}.txt"
        labels = self._load_labels(label_path)

        # Letterbox resize
        img, scale, pad = self._letterbox(img, self.img_size)

        # Adjust labels for letterbox transform
        if labels.numel() > 0:
            labels = self._adjust_labels(labels, scale, pad, orig_w, orig_h)

        # To tensor [C, H, W], float [0, 1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)

        meta = {
            "img_path": str(img_path),
            "orig_shape": (orig_h, orig_w),
            "scale": scale,
            "pad": pad,
        }

        return img_tensor, labels, meta

    def _load_labels(self, label_path: Path) -> torch.Tensor:
        """Load and parse OBB labels from file."""
        if not label_path.exists():
            logger.debug(f"Label file not found: {label_path}")
            return torch.zeros((0, 6), dtype=torch.float32)

        labels_list = []
        with open(label_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) != 9:
                    logger.debug(f"{label_path}:{line_num} invalid format (expected 9 values)")
                    continue

                try:
                    cls = int(parts[0])
                    if self.num_classes is not None and (cls < 0 or cls >= self.num_classes):
                        logger.debug(f"{label_path}:{line_num} class {cls} out of range")
                        continue

                    corners = np.array([float(x) for x in parts[1:9]], dtype=np.float32).reshape(4, 2)
                    cx, cy, w, h, angle = corners_to_xywhr(corners)
                    labels_list.append([cls, cx, cy, w, h, angle])
                except ValueError as e:
                    logger.debug(f"{label_path}:{line_num} parse error: {e}")
                    continue

        if not labels_list:
            return torch.zeros((0, 6), dtype=torch.float32)

        return torch.tensor(labels_list, dtype=torch.float32)

    def _letterbox(
        self, img: np.ndarray, target_size: int
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Resize image with letterbox padding."""
        h, w = img.shape[:2]
        scale = min(target_size / h, target_size / w)

        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = (target_size - new_w) // 2
        pad_h = (target_size - new_h) // 2

        img_padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        img_padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = img_resized

        return img_padded, scale, (pad_w, pad_h)

    def _adjust_labels(
        self,
        labels: torch.Tensor,
        scale: float,
        pad: Tuple[int, int],
        orig_w: int,
        orig_h: int,
    ) -> torch.Tensor:
        """Adjust label coordinates for letterbox transform."""
        labels = labels.clone()
        pad_w, pad_h = pad
        target = self.img_size

        new_w, new_h = orig_w * scale, orig_h * scale

        labels[:, 1] = (labels[:, 1] * new_w + pad_w) / target  # cx
        labels[:, 2] = (labels[:, 2] * new_h + pad_h) / target  # cy
        labels[:, 3] = labels[:, 3] * new_w / target  # w
        labels[:, 4] = labels[:, 4] * new_h / target  # h
        # angle unchanged

        return labels


# =============================================================================
# DATA MODULE
# =============================================================================


class OBBDataModule(pl.LightningDataModule):
    """Lightning DataModule for OBB datasets.

    Handles train/val data loading with proper collation for Ultralytics format.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset (optional).
        batch_size: Batch size per device.
        num_workers: Number of dataloader workers.
    """

    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 8,
        num_workers: int = 4,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    @staticmethod
    def _collate_fn(
        batch: List[Tuple[torch.Tensor, torch.Tensor, Dict]]
    ) -> Dict[str, Any]:
        """Collate batch into Ultralytics-compatible format."""
        imgs = torch.stack([item[0] for item in batch])

        batch_idx_list = []
        cls_list = []
        bbox_list = []
        metas = []

        for img_idx, (_, labels, meta) in enumerate(batch):
            metas.append(meta)
            if labels.numel() == 0:
                continue
            n = labels.shape[0]
            batch_idx_list.append(torch.full((n,), img_idx, dtype=torch.long))
            cls_list.append(labels[:, 0:1])
            bbox_list.append(labels[:, 1:6])

        if not batch_idx_list:
            return {
                "img": imgs,
                "batch_idx": torch.empty(0, dtype=torch.long),
                "cls": torch.empty(0, 1),
                "bboxes": torch.empty(0, 5),
                "meta": metas,
            }

        return {
            "img": imgs,
            "batch_idx": torch.cat(batch_idx_list),
            "cls": torch.cat(cls_list),
            "bboxes": torch.cat(bbox_list),
            "meta": metas,
        }


# =============================================================================
# LIGHTNING MODULE WITH METRICS
# =============================================================================


class YOLOOBBLightning(pl.LightningModule):
    """Lightning module for YOLO-OBB training with Ultralytics metrics.

    Integrates Ultralytics YOLO model, loss function, and OBBMetrics for
    comprehensive training and evaluation.

    Args:
        model_name: Pretrained model name or path.
        num_classes: Number of classes (auto-detected from model if None).
        lr: Learning rate.
        weight_decay: Weight decay coefficient.
        warmup_epochs: Number of warmup epochs.
        img_size: Input image size for metrics computation.
    """

    def __init__(
        self,
        model_name: str = "yolo11n-obb.pt",
        num_classes: Optional[int] = None,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        warmup_epochs: int = 3,
        img_size: int = 640,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._img_size = img_size

        from ultralytics import YOLO

        # Load model
        yolo = YOLO(model_name)
        model_nc = yolo.model.yaml.get("nc", 15)

        # Rebuild model head if num_classes differs
        if num_classes is not None and num_classes != model_nc:
            cfg = yolo.model.yaml.copy()
            cfg["nc"] = num_classes

            # Preserve scale
            if "scale" not in cfg:
                for s in ["n", "s", "m", "l", "x"]:
                    if f"yolo11{s}" in model_name or f"yolov8{s}" in model_name:
                        cfg["scale"] = s
                        break
                else:
                    cfg["scale"] = "n"

            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                yaml.dump(cfg, f)
                temp_yaml = f.name

            try:
                yolo = YOLO(temp_yaml)
            finally:
                Path(temp_yaml).unlink(missing_ok=True)

            logger.debug(f"Rebuilt model head: {model_nc} -> {num_classes} classes")

        self.model = yolo.model
        self.nc = self.model.yaml.get("nc", num_classes or 15)

        # Ensure args exist for loss function
        if not hasattr(self.model, "args") or self.model.args is None:
            self.model.args = {}
        if isinstance(self.model.args, dict):
            defaults = {"box": 7.5, "cls": 0.5, "dfl": 1.5}
            for k, v in defaults.items():
                self.model.args.setdefault(k, v)
            self.model.args = SimpleNamespace(**self.model.args)

        # Initialize loss criterion
        self.criterion = self.model.init_criterion()

        # Initialize OBB metrics
        self._init_metrics()

        logger.info(f"Model: {model_name} ({self.nc} classes, metrics={'enabled' if self._has_metrics else 'disabled'})")

    def _init_metrics(self) -> None:
        """Initialize TorchMetrics MeanAveragePrecision for validation.

        Note: Converts OBB to axis-aligned bounding boxes for TorchMetrics compatibility.
        """
        try:
            from torchmetrics.detection import MeanAveragePrecision
            # Register as module attribute for automatic device placement
            self.map_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
            self._has_metrics = True
        except ImportError:
            logger.warning("torchmetrics[detection] not installed, metrics disabled")
            self.map_metric = None
            self._has_metrics = False

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup model and move metric to device."""
        for param in self.model.parameters():
            param.requires_grad = True

        self.criterion.device = self.device
        if hasattr(self.criterion, "proj"):
            self.criterion.proj = self.criterion.proj.to(self.device)

        if self._has_metrics:
            self.map_metric = self.map_metric.to(self.device)

    def forward(self, x: torch.Tensor) -> Any:
        """Forward pass through YOLO model."""
        return self.model(x)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Execute single training step."""
        imgs = batch["img"]
        preds = self(imgs)

        # Move batch tensors to device
        batch_on_device = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        loss, loss_items = self.criterion(preds, batch_on_device)
        total_loss = loss.sum()

        # Log losses
        self.log("train/loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/box_loss", loss_items[0], on_step=False, on_epoch=True)
        self.log("train/cls_loss", loss_items[1], on_step=False, on_epoch=True)
        self.log("train/dfl_loss", loss_items[2], on_step=False, on_epoch=True)

        return total_loss

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """Execute single validation step with metrics computation."""
        imgs = batch["img"]
        preds = self(imgs)

        # Move batch tensors to device
        batch_on_device = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Compute loss
        loss, loss_items = self.criterion(preds, batch_on_device)
        total_loss = loss.sum()

        # Log losses
        self.log("val/loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("val/box_loss", loss_items[0], sync_dist=True)
        self.log("val/cls_loss", loss_items[1], sync_dist=True)
        self.log("val/dfl_loss", loss_items[2], sync_dist=True)

        # Update metrics if available
        if self._has_metrics:
            try:
                self._update_metrics(preds, batch_on_device)
            except Exception as e:
                # Disable metrics on first failure to avoid spam
                logger.warning(f"Metrics computation failed, disabling: {e}")
                self._has_metrics = False

        return {"loss": total_loss}

    def _update_metrics(self, preds: Any, batch: Dict[str, Any]) -> None:
        """Update TorchMetrics with predictions converted to axis-aligned boxes."""
        from ultralytics.utils.ops import non_max_suppression_rotated

        # Apply NMS to raw predictions
        pred_raw = preds[0] if isinstance(preds, (list, tuple)) else preds
        nms_preds = non_max_suppression_rotated(
            pred_raw, conf_thres=0.001, iou_thres=0.7, nc=self.nc, max_det=300
        )

        batch_idx_tensor = batch["batch_idx"]
        gt_cls_all = batch["cls"]
        gt_bboxes_all = batch["bboxes"]
        scale = self._img_size

        preds_list, targets_list = [], []

        for img_idx, pred in enumerate(nms_preds):
            # Get ground truth for this image
            mask = batch_idx_tensor == img_idx
            gt_cls = gt_cls_all[mask].squeeze(-1) if mask.sum() > 0 else torch.empty(0)
            gt_bboxes = gt_bboxes_all[mask] if mask.sum() > 0 else torch.empty((0, 5))

            # Convert GT OBB (xywhr normalized) to axis-aligned xyxy (pixels)
            if len(gt_bboxes) > 0:
                gt_xyxy = self._obb_to_xyxy(gt_bboxes.to(self.device), scale)
                targets_list.append({
                    "boxes": gt_xyxy,
                    "labels": gt_cls.long().to(self.device),
                })
            else:
                targets_list.append({
                    "boxes": torch.empty((0, 4), device=self.device),
                    "labels": torch.empty(0, dtype=torch.long, device=self.device),
                })

            # Convert predictions OBB to axis-aligned xyxy
            if pred is not None and len(pred) > 0:
                pred_xyxy = self._obb_to_xyxy(pred[:, :5], scale)
                preds_list.append({
                    "boxes": pred_xyxy,
                    "scores": pred[:, 5],
                    "labels": pred[:, 6].long(),
                })
            else:
                preds_list.append({
                    "boxes": torch.empty((0, 4), device=self.device),
                    "scores": torch.empty(0, device=self.device),
                    "labels": torch.empty(0, dtype=torch.long, device=self.device),
                })

        self.map_metric.update(preds_list, targets_list)

    @staticmethod
    def _obb_to_xyxy(obb: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """Convert OBB (xywhr) to axis-aligned bounding box (xyxy).

        Args:
            obb: Oriented boxes [N, 5] as (cx, cy, w, h, angle)
            scale: Scale factor (for normalized coords to pixels)

        Returns:
            Axis-aligned boxes [N, 4] as (x1, y1, x2, y2)
        """
        if len(obb) == 0:
            return torch.empty((0, 4), device=obb.device)

        cx, cy, w, h, angle = obb[:, 0], obb[:, 1], obb[:, 2], obb[:, 3], obb[:, 4]

        # Scale to pixels
        cx, cy, w, h = cx * scale, cy * scale, w * scale, h * scale

        # Compute corners of rotated box
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)

        # Half dimensions
        hw, hh = w / 2, h / 2

        # Four corners relative to center
        dx = torch.stack([hw, hw, -hw, -hw], dim=1)
        dy = torch.stack([hh, -hh, -hh, hh], dim=1)

        # Rotate corners
        rx = dx * cos_a.unsqueeze(1) - dy * sin_a.unsqueeze(1)
        ry = dx * sin_a.unsqueeze(1) + dy * cos_a.unsqueeze(1)

        # Absolute corner positions
        corners_x = cx.unsqueeze(1) + rx
        corners_y = cy.unsqueeze(1) + ry

        # Get axis-aligned bounding box
        x1 = corners_x.min(dim=1).values
        y1 = corners_y.min(dim=1).values
        x2 = corners_x.max(dim=1).values
        y2 = corners_y.max(dim=1).values

        return torch.stack([x1, y1, x2, y2], dim=1)

    def on_validation_epoch_start(self) -> None:
        """Reset metrics at start of validation epoch."""
        if self._has_metrics:
            self.map_metric.reset()

    def on_validation_epoch_end(self) -> None:
        """Compute and log final validation metrics."""
        if not self._has_metrics:
            return

        try:
            metrics = self.map_metric.compute()
            map50 = metrics["map_50"].item()
            map50_95 = metrics["map"].item()

            self.log("val/mAP50", map50, sync_dist=True)
            self.log("val/mAP50-95", map50_95, sync_dist=True)
            logger.info(f"Validation: mAP50={map50:.4f}, mAP50-95={map50_95:.4f}")
        except Exception as e:
            logger.warning(f"Failed to compute metrics: {e}")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        # Separate parameters into decay and no_decay groups
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bn" in name or "bias" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        logger.debug(f"Optimizer groups: {len(decay)} decay, {len(no_decay)} no-decay params")

        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.hparams.lr,
        )

        # Cosine annealing scheduler
        total_epochs = self.trainer.max_epochs
        warmup = self.hparams.warmup_epochs

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs - warmup),
            eta_min=self.hparams.lr * 0.01,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def on_train_epoch_start(self) -> None:
        """Apply linear warmup during initial epochs."""
        if self.current_epoch < self.hparams.warmup_epochs:
            factor = (self.current_epoch + 1) / self.hparams.warmup_epochs
            for pg in self.optimizers().param_groups:
                pg["lr"] = self.hparams.lr * factor
            logger.debug(f"Warmup epoch {self.current_epoch + 1}: lr={self.hparams.lr * factor:.2e}")


# =============================================================================
# TRAINING FUNCTION
# =============================================================================


def train(
    data: str,
    model: str = "yolo11n-obb.pt",
    epochs: int = 100,
    batch_size: int = 6,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    warmup_epochs: int = 3,
    img_size: int = 640,
    workers: int = 4,
    num_classes: Optional[int] = None,
    precision: str = "16-mixed",
    gradient_clip_val: float = 10.0,
    val_check_interval: float = 1.0,
    log_every_n_steps: int = 10,
    save_top_k: int = 3,
    output_dir: str = "./outputs",
) -> None:
    """Execute YOLO-OBB training with PyTorch Lightning.

    This is the main entry point for training. It sets up datasets,
    model, and trainer, then runs the training loop.

    Args:
        data: Path to dataset root directory (must contain images/ and labels/).
        model: Pretrained YOLO-OBB model name or path (e.g., 'yolo11n-obb.pt').
        epochs: Total number of training epochs.
        batch_size: Number of samples per batch.
        lr: Initial learning rate for AdamW optimizer.
        weight_decay: L2 regularization coefficient.
        warmup_epochs: Number of linear warmup epochs.
        img_size: Input image size (square, e.g., 640).
        workers: Number of dataloader worker processes.
        num_classes: Override number of classes (auto-detected if None).
        precision: Training precision ('16-mixed', '32', 'bf16-mixed').
        gradient_clip_val: Max gradient norm for clipping (0 to disable).
        val_check_interval: Validation frequency (1.0 = every epoch).
        log_every_n_steps: Logging frequency in training steps.
        save_top_k: Number of best checkpoints to save.
        output_dir: Directory for checkpoints and logs.
    """
    # Detect number of classes
    if num_classes is None:
        num_classes = detect_num_classes(data)

    # Create datasets
    train_ds = YOLOOBBDataset(data, "train", img_size, num_classes)
    val_ds = YOLOOBBDataset(data, "val", img_size, num_classes)

    datamodule = OBBDataModule(train_ds, val_ds, batch_size, workers)

    # Create model
    lightning_model = YOLOOBBLightning(
        model_name=model,
        num_classes=num_classes,
        lr=lr,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        img_size=img_size,
    )

    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_path / "checkpoints",
            filename="yolo-obb-{epoch:03d}-{val_loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=save_top_k,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Determine precision
    train_precision = precision
    if not torch.cuda.is_available() and "16" in precision:
        logger.warning("CUDA not available, falling back to 32-bit precision")
        train_precision = 32

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        precision=train_precision,
        callbacks=callbacks,
        default_root_dir=str(output_path),
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
        gradient_clip_val=gradient_clip_val,
        enable_progress_bar=True,
    )

    logger.info(
        f"YOLO-OBB Training with PyTorch Lightning (v4)\n"
        f"{'=' * 50}\n"
        f"Configuration:\n"
        f"  Data:       {data}\n"
        f"  Model:      {model}\n"
        f"  Classes:    {num_classes}\n"
        f"  Epochs:     {epochs}\n"
        f"  Batch size: {batch_size}\n"
        f"  LR:         {lr}\n"
        f"  Image size: {img_size}\n"
        f"  Precision:  {train_precision}\n"
        f"  Output:     {output_path}"
    )

    # Start training
    trainer.fit(lightning_model, datamodule=datamodule)

    logger.info(f"Training complete! Checkpoints saved to: {output_path / 'checkpoints'}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


def main() -> None:
    """Main entry point using jsonargparse for CLI configuration.

    Supports:
        - Command line arguments: --data /path --epochs 100
        - Config file: --config config.yaml
        - Environment variables: TRAIN_DATA=/path
    """
    try:
        from jsonargparse import CLI
    except ImportError:
        logger.error(
            "jsonargparse not installed. Install with: pip install 'jsonargparse[signatures]'"
        )
        sys.exit(1)

    # Use jsonargparse CLI which automatically:
    # - Parses function arguments from type hints
    # - Extracts help text from docstrings
    # - Supports config files (--config file.yaml)
    # - Supports environment variables
    CLI(train, as_positional=False)


if __name__ == "__main__":
    main()