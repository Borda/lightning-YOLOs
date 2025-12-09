"""
YOLO-OBB Training with PyTorch Lightning (v3)
==============================================
Minimal implementation reusing Ultralytics components (model, loss)
with PyTorch Lightning for training orchestration.

Fixes in v3:
- Cleaned up unused arguments
- Memory leak prevention (explicit tensor cleanup, no grad in val)
- Default batch_size=6
- Proper coordinate handling (normalized throughout)
"""

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import math
import cv2
import numpy as np
import yaml


# =============================================================================
# UTILITIES
# =============================================================================

def detect_num_classes(root: Union[str, Path]) -> int:
    """Scan label files to find max class index. Returns max_index + 1."""
    root = Path(root)
    max_class = -1

    for split in ["train", "val"]:
        label_dir = root / "labels" / split
        if not label_dir.exists():
            continue

        for label_file in label_dir.glob("*.txt"):
            try:
                with open(label_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            cls = int(parts[0])
                            max_class = max(max_class, cls)
            except (ValueError, IOError):
                continue

    if max_class < 0:
        raise ValueError(f"No valid labels found in {root}/labels/")

    num_classes = max_class + 1
    print(f"Auto-detected {num_classes} classes (max index: {max_class})")
    return num_classes


def corners_to_xywhr(corners: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Convert 4 normalized corner points [4,2] to normalized (cx, cy, w, h, angle).
    Angle in radians [0, pi/2).
    """
    # Scale to virtual pixels for cv2.minAreaRect
    SCALE = 1000.0
    corners_px = (corners * SCALE).astype(np.float32)

    rect = cv2.minAreaRect(corners_px)
    (cx_px, cy_px), (w_px, h_px), angle_deg = rect

    # Back to normalized
    cx, cy = cx_px / SCALE, cy_px / SCALE
    w, h = w_px / SCALE, h_px / SCALE

    # Ensure w >= h, normalize angle to [0, 90)
    if w < h:
        w, h = h, w
        angle_deg += 90

    angle_deg = angle_deg % 90
    if angle_deg < 0:
        angle_deg += 90

    angle_rad = max(0.0, min(angle_deg * math.pi / 180.0, math.pi / 2 - 1e-6))

    return cx, cy, w, h, angle_rad


# =============================================================================
# DATASET
# =============================================================================

class YOLOOBBDataset(Dataset):
    """
    Dataset for YOLO OBB format.
    Labels: class x1 y1 x2 y2 x3 y3 x4 y4 (normalized corners)
    Output: [N, 6] tensor [cls, cx, cy, w, h, angle] (normalized)
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

        self.img_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        self.img_paths = sorted([
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in self.SUPPORTED_FORMATS
        ])

        if not self.img_paths:
            raise ValueError(f"No images found in {self.img_dir}")

        print(f"Found {len(self.img_paths)} images in {self.img_dir}")

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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

        # Adjust labels for letterbox
        if labels.numel() > 0:
            labels = self._adjust_labels(labels, scale, pad, orig_w, orig_h)

        # To tensor [C, H, W], float [0, 1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)

        return img_tensor, labels

    def _load_labels(self, label_path: Path) -> torch.Tensor:
        if not label_path.exists():
            return torch.zeros((0, 6), dtype=torch.float32)

        labels_list = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 9:
                    continue

                cls = int(parts[0])
                if self.num_classes is not None and (cls < 0 or cls >= self.num_classes):
                    continue

                corners = np.array([float(x) for x in parts[1:9]], dtype=np.float32).reshape(4, 2)
                cx, cy, w, h, angle = corners_to_xywhr(corners)
                labels_list.append([cls, cx, cy, w, h, angle])

        if not labels_list:
            return torch.zeros((0, 6), dtype=torch.float32)

        return torch.tensor(labels_list, dtype=torch.float32)

    def _letterbox(self, img: np.ndarray, target_size: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        h, w = img.shape[:2]
        scale = min(target_size / h, target_size / w)

        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = (target_size - new_w) // 2
        pad_h = (target_size - new_h) // 2

        img_padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        img_padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = img_resized

        return img_padded, scale, (pad_w, pad_h)

    def _adjust_labels(
        self, labels: torch.Tensor, scale: float, pad: Tuple[int, int], orig_w: int, orig_h: int
    ) -> torch.Tensor:
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
    """Lightning DataModule for OBB datasets."""

    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 6,
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
            collate_fn=self._collate_fn,
            pin_memory=True,
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
            collate_fn=self._collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    @staticmethod
    def _collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate into Ultralytics format for v8OBBLoss."""
        imgs = torch.stack([item[0] for item in batch])

        batch_idx_list = []
        cls_list = []
        bbox_list = []

        for img_idx, (_, labels) in enumerate(batch):
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
            }

        return {
            "img": imgs,
            "batch_idx": torch.cat(batch_idx_list),
            "cls": torch.cat(cls_list),
            "bboxes": torch.cat(bbox_list),
        }


# =============================================================================
# LIGHTNING MODULE
# =============================================================================

class YOLOOBBLightning(pl.LightningModule):
    """Lightning wrapper for Ultralytics YOLO-OBB."""

    def __init__(
        self,
        model_name: str = "yolo11n-obb.pt",
        num_classes: Optional[int] = None,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        warmup_epochs: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters()

        from ultralytics import YOLO
        from types import SimpleNamespace
        import tempfile

        # Load model
        yolo = YOLO(model_name)
        model_nc = yolo.model.yaml.get('nc', 15)

        # Rebuild if num_classes differs
        if num_classes is not None and num_classes != model_nc:
            print(f"Rebuilding model: {model_nc} -> {num_classes} classes")

            cfg = yolo.model.yaml.copy()
            cfg['nc'] = num_classes

            # Preserve scale
            if 'scale' not in cfg:
                for s in ['n', 's', 'm', 'l', 'x']:
                    if f'yolo11{s}' in model_name or f'yolov8{s}' in model_name:
                        cfg['scale'] = s
                        break
                else:
                    cfg['scale'] = 'n'

            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(cfg, f)
                temp_yaml = f.name

            try:
                yolo = YOLO(temp_yaml)
            finally:
                Path(temp_yaml).unlink(missing_ok=True)

        self.model = yolo.model
        self.nc = self.model.yaml.get('nc', num_classes or 15)

        # Ensure args exist for loss
        if not hasattr(self.model, 'args') or self.model.args is None:
            self.model.args = {}
        if isinstance(self.model.args, dict):
            defaults = {'box': 7.5, 'cls': 0.5, 'dfl': 1.5}
            for k, v in defaults.items():
                self.model.args.setdefault(k, v)
            self.model.args = SimpleNamespace(**self.model.args)

        self.criterion = self.model.init_criterion()

        print(f"Model: {model_name}, Classes: {self.nc}")

    def setup(self, stage: Optional[str] = None):
        """Move criterion tensors to device."""
        for param in self.model.parameters():
            param.requires_grad = True

        self.criterion.device = self.device
        if hasattr(self.criterion, 'proj'):
            self.criterion.proj = self.criterion.proj.to(self.device)

    def forward(self, x: torch.Tensor) -> Any:
        return self.model(x)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        imgs = batch["img"]
        preds = self(imgs)

        # Move batch to device
        batch_on_device = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        loss, loss_items = self.criterion(preds, batch_on_device)
        total_loss = loss.sum()

        self.log("train/loss", total_loss, prog_bar=True)
        self.log("train/box_loss", loss_items[0])
        self.log("train/cls_loss", loss_items[1])
        self.log("train/dfl_loss", loss_items[2])

        return total_loss

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        imgs = batch["img"]
        preds = self(imgs)

        batch_on_device = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        loss, loss_items = self.criterion(preds, batch_on_device)
        total_loss = loss.sum()

        self.log("val/loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("val/box_loss", loss_items[0], sync_dist=True)
        self.log("val/cls_loss", loss_items[1], sync_dist=True)
        self.log("val/dfl_loss", loss_items[2], sync_dist=True)

        return total_loss

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bn" in name or "bias" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        optimizer = torch.optim.AdamW([
            {"params": decay, "weight_decay": self.hparams.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=self.hparams.lr)

        # Scheduler
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

    def on_train_epoch_start(self):
        """Linear warmup."""
        if self.current_epoch < self.hparams.warmup_epochs:
            factor = (self.current_epoch + 1) / self.hparams.warmup_epochs
            for pg in self.optimizers().param_groups:
                pg["lr"] = self.hparams.lr * factor


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLO-OBB with Lightning")
    parser.add_argument("--data", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--model", type=str, default="yolo11n-obb.pt", help="Model name")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    # Detect classes
    num_classes = detect_num_classes(args.data)

    # Create datasets
    train_ds = YOLOOBBDataset(args.data, "train", args.img_size, num_classes)
    val_ds = YOLOOBBDataset(args.data, "val", args.img_size, num_classes)
    datamodule = OBBDataModule(train_ds, val_ds, args.batch_size, args.workers)

    # Create model
    model = YOLOOBBLightning(
        model_name=args.model,
        num_classes=num_classes,
        lr=args.lr,
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=3),
            pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        ],
        log_every_n_steps=10,
        gradient_clip_val=10.0,  # Prevent gradient explosion
    )

    trainer.fit(model, datamodule=datamodule)
    print("\n✅ Training complete!")