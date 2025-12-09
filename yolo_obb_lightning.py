"""
YOLO-OBB Training with PyTorch Lightning (v6)

Features: logging, jsonargparse CLI, TorchMetrics mAP (train+val), AMP, auto class detection.

Usage:
    python yolo_obb_lightning_v6.py --data /path/to/dataset --model yolo11n-obb.pt
    python yolo_obb_lightning_v6.py --config config.yaml

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
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
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
                for line in open(lf):
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
        for line in open(path):
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


# =============================================================================
# LIGHTNING MODULE
# =============================================================================


class YOLOOBBLightning(pl.LightningModule):
    """Lightning module for YOLO-OBB training with TorchMetrics."""

    def __init__(
        self,
        model_name: str = "yolo11n-obb.pt",
        num_classes: int = 15,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        warmup_epochs: int = 3,
        img_size: int = 640,
    ):
        super().__init__()
        self.save_hyperparameters()

        from ultralytics import YOLO

        yolo = YOLO(model_name)
        model_nc = yolo.model.yaml.get("nc", 15)

        if num_classes != model_nc:
            cfg = {**yolo.model.yaml, "nc": num_classes}
            if "scale" not in cfg:
                cfg["scale"] = next((s for s in "nsmlx" if f"yolo11{s}" in model_name or f"yolov8{s}" in model_name), "n")

            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                yaml.dump(cfg, f)
                temp_yaml = f.name
            try:
                yolo = YOLO(temp_yaml)
            finally:
                Path(temp_yaml).unlink(missing_ok=True)
            logger.debug(f"Rebuilt model: {model_nc} -> {num_classes} classes")

        self.model = yolo.model
        self.nc = num_classes

        if not hasattr(self.model, "args") or self.model.args is None:
            self.model.args = {}
        if isinstance(self.model.args, dict):
            self.model.args = SimpleNamespace(**{**{"box": 7.5, "cls": 0.5, "dfl": 1.5}, **self.model.args})

        self.criterion = self.model.init_criterion()

        # Initialize metrics (train and val)
        try:
            from torchmetrics.detection import MeanAveragePrecision
            self.train_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
            self.val_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        except ImportError:
            logger.warning("torchmetrics[detection] not installed, metrics disabled")
            self.train_map = self.val_map = None

        logger.info(f"Model: {model_name} ({self.nc} classes, metrics={'on' if self.val_map else 'off'})")

    def setup(self, stage: str | None = None):
        for p in self.model.parameters():
            p.requires_grad = True
        self.criterion.device = self.device
        if hasattr(self.criterion, "proj"):
            self.criterion.proj = self.criterion.proj.to(self.device)
        if self.val_map:
            self.train_map = self.train_map.to(self.device)
            self.val_map = self.val_map.to(self.device)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def _to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def _compute_step(self, batch: dict, stage: str) -> torch.Tensor:
        """Shared logic for training and validation steps."""
        preds = self(batch["img"])
        batch_dev = self._to_device(batch)
        loss, items = self.criterion(preds, batch_dev)
        total = loss.sum()

        self.log(f"{stage}/loss", total, prog_bar=True, on_step=(stage == "train"), on_epoch=True, sync_dist=(stage == "val"))
        self.log_dict({f"{stage}/box": items[0], f"{stage}/cls": items[1], f"{stage}/dfl": items[2]}, on_epoch=True, sync_dist=(stage == "val"))

        # Update metrics
        metric = self.train_map if stage == "train" else self.val_map
        if metric is not None:
            try:
                self._update_metrics(preds, batch_dev, metric)
            except Exception as e:
                logger.warning(f"{stage} metrics failed: {e}")

        return total

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._compute_step(batch, "train")

    @torch.no_grad()
    def validation_step(self, batch: dict, batch_idx: int):
        return self._compute_step(batch, "val")

    def _update_metrics(self, preds: Any, batch: dict, metric):
        from ultralytics.utils.nms import non_max_suppression

        raw = preds[0] if isinstance(preds, (list, tuple)) else preds
        nms_preds = non_max_suppression(raw, conf_thres=0.001, iou_thres=0.7, nc=self.nc, max_det=300, rotated=True)

        preds_list, targets_list = [], []
        img_size = self.hparams.img_size

        for i, pred in enumerate(nms_preds):
            mask = batch["batch_idx"] == i
            gt_cls = batch["cls"][mask].squeeze(-1) if mask.sum() else torch.empty(0, device=self.device)
            gt_box = batch["bboxes"][mask] if mask.sum() else torch.empty((0, 5), device=self.device)

            targets_list.append({
                "boxes": obb_to_xyxy(gt_box, img_size) if len(gt_box) else torch.empty((0, 4), device=self.device),
                "labels": gt_cls.long(),
            })

            has_pred = pred is not None and len(pred)
            preds_list.append({
                "boxes": obb_to_xyxy(pred[:, :5], img_size) if has_pred else torch.empty((0, 4), device=self.device),
                "scores": pred[:, 5] if has_pred else torch.empty(0, device=self.device),
                "labels": pred[:, 6].long() if has_pred else torch.empty(0, dtype=torch.long, device=self.device),
            })

        metric.update(preds_list, targets_list)

    def _log_metrics(self, stage: str):
        metric = self.train_map if stage == "train" else self.val_map
        if metric is None:
            return
        try:
            m = metric.compute()
            self.log(f"{stage}/mAP50", m["map_50"], sync_dist=True)
            self.log(f"{stage}/mAP50-95", m["map"], sync_dist=True)
            logger.info(f"{stage.capitalize()}: mAP50={m['map_50']:.4f}, mAP50-95={m['map']:.4f}")
            metric.reset()
        except Exception as e:
            logger.warning(f"Failed to compute {stage} metrics: {e}")

    def on_train_epoch_end(self):
        self._log_metrics("train")

    def on_validation_epoch_end(self):
        self._log_metrics("val")

    def configure_optimizers(self):
        decay = [p for n, p in self.model.named_parameters() if p.requires_grad and "bn" not in n and "bias" not in n]
        no_decay = [p for n, p in self.model.named_parameters() if p.requires_grad and ("bn" in n or "bias" in n)]

        optimizer = torch.optim.AdamW([
            {"params": decay, "weight_decay": self.hparams.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.trainer.max_epochs - self.hparams.warmup_epochs),
            eta_min=self.hparams.lr * 0.01
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def on_train_epoch_start(self):
        if self.current_epoch < self.hparams.warmup_epochs:
            factor = (self.current_epoch + 1) / self.hparams.warmup_epochs
            for pg in self.optimizers().param_groups:
                pg["lr"] = self.hparams.lr * factor


# =============================================================================
# TRAINING
# =============================================================================


def train(
    data: str,
    model: str = "yolo11n-obb.pt",
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    warmup_epochs: int = 3,
    img_size: int = 640,
    workers: int = 4,
    num_classes: int | None = None,
    precision: str = "16-mixed",
    gradient_clip_val: float = 10.0,
    val_check_interval: float = 1.0,
    log_every_n_steps: int = 10,
    save_top_k: int = 3,
    output_dir: str = "./outputs",
) -> None:
    """Train YOLO-OBB model with PyTorch Lightning.

    Args:
        data: Dataset root (must contain images/ and labels/).
        model: Pretrained YOLO-OBB model.
        epochs: Training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        weight_decay: L2 regularization.
        warmup_epochs: Warmup epochs.
        img_size: Input size.
        workers: Dataloader workers.
        num_classes: Override class count (auto-detected if None).
        precision: Training precision.
        gradient_clip_val: Gradient clipping.
        val_check_interval: Validation frequency.
        log_every_n_steps: Logging frequency.
        save_top_k: Checkpoints to keep.
        output_dir: Output directory.
    """
    # DataModule handles dataset creation and class detection
    dm = OBBDataModule(data, img_size, batch_size, workers, num_classes)
    nc = dm.num_classes  # Triggers auto-detection if needed

    lightning_model = YOLOOBBLightning(model, nc, lr, weight_decay, warmup_epochs, img_size)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    prec = precision if torch.cuda.is_available() or "16" not in precision else 32
    if prec != precision:
        logger.warning("CUDA unavailable, using 32-bit precision")

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        precision=prec,
        callbacks=[
            ModelCheckpoint(dirpath=output / "checkpoints", filename="yolo-obb-{epoch:03d}-{val_loss:.4f}",
                            monitor="val/loss", mode="min", save_top_k=save_top_k, save_last=True),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        default_root_dir=str(output),
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
        gradient_clip_val=gradient_clip_val,
    )

    logger.info(
        f"YOLO-OBB Training (v6)\n{'=' * 40}\n"
        f"Data: {data} | Model: {model} | Classes: {nc}\n"
        f"Epochs: {epochs} | Batch: {batch_size} | LR: {lr} | Size: {img_size}"
    )

    trainer.fit(lightning_model, datamodule=dm)
    logger.info(f"Done! Checkpoints: {output / 'checkpoints'}")


def main():
    try:
        from jsonargparse import CLI
    except ImportError:
        logger.error("Install jsonargparse: pip install 'jsonargparse[signatures]'")
        sys.exit(1)
    CLI(train, as_positional=False)


if __name__ == "__main__":
    main()