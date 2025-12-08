"""
YOLO-OBB Training with PyTorch Lightning
=========================================
Minimal implementation that reuses Ultralytics components (model, loss)
with PyTorch Lightning for training orchestration.

Design goals:
- Minimal code duplication
- Reuse battle-tested Ultralytics loss (v8OBBLoss)
- Proper Lightning DataModule for data handling
- Compatible with standard YOLO OBB dataset format

Dataset structure (YOLO OBB format):
    dataset/
    ├── images/
    │   ├── train/
    │   │   ├── img1.jpg
    │   │   └── ...
    │   └── val/
    │       └── ...
    └── labels/
        ├── train/
        │   ├── img1.txt  # class x1 y1 x2 y2 x3 y3 x4 y4 (4 corners, normalized)
        │   └── ...
        └── val/
            └── ...
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
import logging


# =============================================================================
# YOLO OBB DATASET
# =============================================================================

class YOLOOBBDataset(Dataset):
    """
    Dataset for YOLO OBB format.
    
    Label format in .txt files: class x1 y1 x2 y2 x3 y3 x4 y4
    - 4 corner points, normalized to [0, 1]
    - Converted internally to xywhr format for loss computation
    
    Directory structure:
        root/
        ├── images/{split}/  (jpg, png, etc.)
        └── labels/{split}/  (.txt files, same names as images)
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
        self.split = split
        self.img_size = img_size
        self.num_classes = num_classes
        self._invalid_label_files: set[str] = set()
        self._invalid_label_total = 0

        # Find image directory
        self.img_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split
        
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        
        # Collect image paths
        self.img_paths = sorted([
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in self.SUPPORTED_FORMATS
        ])
        
        if len(self.img_paths) == 0:
            raise ValueError(f"No images found in {self.img_dir}")
        
        logger = logging.getLogger(__name__)
        logger.info(f"Found {len(self.img_paths)} images in {self.img_dir}")
    
    def __len__(self) -> int:
        return len(self.img_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        img_path = self.img_paths[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        orig_h, orig_w = img.shape[:2]
        
        # Load labels
        label_path = self.label_dir / f"{img_path.stem}.txt"
        labels = self._load_labels(label_path, orig_w, orig_h)
        
        # Resize image (letterbox to maintain aspect ratio)
        img, scale, pad = self._letterbox(img, self.img_size)
        
        # Adjust label coordinates for letterbox
        if labels.numel() > 0:
            labels = self._adjust_labels_for_letterbox(labels, scale, pad, orig_w, orig_h)

        # Convert to tensor [C, H, W], normalize to [0, 1]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        return img, labels
    
    def _load_labels(self, label_path: Path, img_w: int, img_h: int) -> torch.Tensor:
        """
        Load OBB labels from txt file.
        
        Input format: class x1 y1 x2 y2 x3 y3 x4 y4 (normalized corners)
        Output format: [N, 6] tensor with [class, cx, cy, w, h, angle]
        
        Uses Ultralytics' xyxyxyxy2xywhr for conversion to ensure consistency
        with their loss function expectations.
        """
        if not label_path.exists():
            return torch.zeros((0, 6))
        
        classes = []
        corners_list = []
        
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 9:  # class + 8 coordinates
                    continue
                
                classes.append(int(parts[0]))
                # 4 corner points as flat array [x1,y1,x2,y2,x3,y3,x4,y4]
                corners_list.append([float(x) for x in parts[1:9]])
        
        if not corners_list:
            return torch.zeros((0, 6))
        
        # Use Ultralytics' conversion function for consistency
        from ultralytics.utils.ops import xyxyxyxy2xywhr
        
        corners = np.array(corners_list, dtype=np.float32)  # [N, 8]
        xywhr = xyxyxyxy2xywhr(corners)  # [N, 5] - uses cv2.minAreaRect internally
        
        # Combine class and xywhr
        classes = np.array(classes, dtype=np.float32).reshape(-1, 1)
        labels = np.hstack([classes, xywhr])
        labels = torch.from_numpy(labels).float()

        if self.num_classes is not None and labels.numel() > 0:
            cls_col = labels[:, 0]
            valid_mask = (cls_col >= 0) & (cls_col < self.num_classes)
            if not valid_mask.all():
                dropped = int((~valid_mask).sum().item())
                self._invalid_label_total += dropped
                if label_path.name not in self._invalid_label_files:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"{label_path.name}: dropping {dropped} labels with class outside [0, {self.num_classes - 1}]")
                    self._invalid_label_files.add(label_path.name)
                labels = labels[valid_mask]

        return labels

    @staticmethod
    def _letterbox(
        img: np.ndarray, 
        target_size: int
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Resize image with letterboxing to maintain aspect ratio."""
        h, w = img.shape[:2]
        scale = min(target_size / h, target_size / w)
        
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size
        pad_w = (target_size - new_w) // 2
        pad_h = (target_size - new_h) // 2
        
        img_padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        img_padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = img_resized
        
        return img_padded, scale, (pad_w, pad_h)
    
    def _adjust_labels_for_letterbox(
        self,
        labels: torch.Tensor,
        scale: float,
        pad: Tuple[int, int],
        orig_w: int,
        orig_h: int,
    ) -> torch.Tensor:
        """Adjust normalized labels for letterbox transformation."""
        # labels: [N, 6] with [cls, cx, cy, w, h, angle]
        labels = labels.clone()
        
        pad_w, pad_h = pad
        target_size = self.img_size

        # Adjust center coordinates
        labels[:, 1] = (labels[:, 1] * orig_w * scale + pad_w) / target_size
        labels[:, 2] = (labels[:, 2] * orig_h * scale + pad_h) / target_size

        # Adjust width/height
        labels[:, 3] = labels[:, 3] * orig_w * scale / target_size
        labels[:, 4] = labels[:, 4] * orig_h * scale / target_size

        # Angle unchanged
        return labels


# =============================================================================
# DATA MODULE
# =============================================================================

class OBBDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for OBB datasets.
    
    Handles batch collation in Ultralytics format expected by v8OBBLoss:
    - batch_idx: [N] image index for each target
    - cls: [N, 1] class labels
    - bboxes: [N, 5] as xywhr (normalized coords, angle in radians)
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
    
    @classmethod
    def from_yaml(
        cls,
        yaml_path: Union[str, Path],
        img_size: int = 640,
        batch_size: int = 8,
        num_workers: int = 4,
        num_classes: Optional[int] = None,
    ) -> "OBBDataModule":
        """
        Create DataModule from YOLO dataset YAML config.
        
        YAML format:
            path: /path/to/dataset
            train: images/train
            val: images/val
            names:
              0: class1
              1: class2
        """
        yaml_path = Path(yaml_path)
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        
        root = Path(config["path"])
        if not root.is_absolute():
            root = yaml_path.parent / root
        
        train_ds = YOLOOBBDataset(root, split="train", img_size=img_size, num_classes=num_classes)
        val_ds = YOLOOBBDataset(root, split="val", img_size=img_size, num_classes=num_classes)

        return cls(train_ds, val_ds, batch_size, num_workers)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=True,
        )
    
    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
    
    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch into Ultralytics format for v8OBBLoss.
        
        Input: List of (image, labels) where labels is [M, 6]: [cls, x, y, w, h, angle]
        Output: Dict with 'img', 'batch_idx', 'cls', 'bboxes'
        """
        imgs = torch.stack([item[0] for item in batch])
        
        batch_idx = []
        cls_list = []
        bbox_list = []
        
        for img_idx, (_, labels) in enumerate(batch):
            if labels.numel() == 0:
                continue
            n = labels.shape[0]
            batch_idx.append(torch.full((n,), img_idx, dtype=torch.long))
            cls_list.append(labels[:, 0:1])   # class
            bbox_list.append(labels[:, 1:6])  # x, y, w, h, angle
        
        # Handle empty batch
        if not batch_idx:
            return {
                "img": imgs,
                "batch_idx": torch.empty(0, dtype=torch.long),
                "cls": torch.empty(0, 1),
                "bboxes": torch.empty(0, 5),
            }
        
        return {
            "img": imgs,
            "batch_idx": torch.cat(batch_idx),
            "cls": torch.cat(cls_list),
            "bboxes": torch.cat(bbox_list),
        }


# =============================================================================
# LIGHTNING MODULE
# =============================================================================

class YOLOOBBLightning(pl.LightningModule):
    """
    Minimal Lightning wrapper for Ultralytics YOLO-OBB.
    
    Reuses:
    - OBBModel (the nn.Module inside YOLO)
    - v8OBBLoss (initialized via model.init_criterion())
    """
    
    def __init__(
        self,
        model_name: str = "yolo11n-obb.pt",
        lr: float = 1e-4,
        weight_decay: float = 5e-4,
        warmup_epochs: int = 3,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        from ultralytics import YOLO
        from types import SimpleNamespace

        # Load Ultralytics model and extract the inner nn.Module
        yolo = YOLO(model_name)
        
        # Override number of classes if specified
        if num_classes is not None:
            yolo.model.yaml['nc'] = num_classes
        
        self.model = yolo.model  # OBBModel - standard nn.Module
        
        # Ensure model.args has hyperparameters as attributes (needed by loss function)
        if isinstance(self.model.args, dict):
            # Add default hyperparameters if missing
            hyp_defaults = {'box': 7.5, 'cls': 0.5, 'dfl': 1.5}
            for key, val in hyp_defaults.items():
                if key not in self.model.args:
                    self.model.args[key] = val
            # Convert dict to namespace so loss can access via dot notation
            self.model.args = SimpleNamespace(**self.model.args)

        # Get loss from Ultralytics (v8OBBLoss for OBB models)
        self.criterion = self.model.init_criterion()
        
        # Store useful attributes
        self.nc = self.model.nc
        self.names = self.model.names
    
    def setup(self, stage: Optional[str] = None):
        """Called after model is moved to device. Update criterion device and enable gradients."""
        # Enable gradients for all parameters (they might be frozen from pretrained model)
        for param in self.model.parameters():
            param.requires_grad = True

        # Update criterion's device attribute to match model's device
        self.criterion.device = self.device

        # Move criterion's internal tensors to the correct device
        if hasattr(self.criterion, 'proj'):
            self.criterion.proj = self.criterion.proj.to(self.device)
        if hasattr(self.criterion, 'bbox_loss'):
            self.criterion.bbox_loss = self.criterion.bbox_loss.to(self.device)

    def forward(self, x: torch.Tensor) -> Any:
        """Forward pass returns raw predictions for loss computation."""
        return self.model(x)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # Ensure model is in training mode (YOLO models have special BN handling)
        # self.model.train()

        imgs = batch["img"]
        preds = self(imgs)
        
        # Move batch tensors to the same device as the model
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        loss, loss_items = self.criterion(preds, batch)
        
        # loss is a tensor [box_loss, cls_loss, dfl_loss], sum to get total
        total_loss = loss.sum()

        self.log("train/loss", total_loss, prog_bar=True)
        self.log("train/box_loss", loss_items[0])
        self.log("train/cls_loss", loss_items[1])
        self.log("train/dfl_loss", loss_items[2])
        
        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        imgs = batch["img"]
        preds = self(imgs)
        
        # Move batch tensors to the same device as the model
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        loss, loss_items = self.criterion(preds, batch)
        
        # loss is a tensor [box_loss, cls_loss, dfl_loss], sum to get total
        total_loss = loss.sum()

        self.log("val/loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("val/box_loss", loss_items[0], sync_dist=True)
        self.log("val/cls_loss", loss_items[1], sync_dist=True)
        self.log("val/dfl_loss", loss_items[2], sync_dist=True)
        
        return total_loss

    def configure_optimizers(self):
        # Separate params: no weight decay for bias and batchnorm
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            (no_decay if "bn" in name or "bias" in name else decay).append(param)
        
        optimizer = torch.optim.AdamW([
            {"params": decay, "weight_decay": self.hparams.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=self.hparams.lr)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs - self.hparams.warmup_epochs,
            eta_min=self.hparams.lr * 0.01,
        )
        
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
    
    def on_train_epoch_start(self):
        """Linear warmup and ensure model is in train mode."""
        # Ensure model is in training mode
        # self.model.train()

        if self.current_epoch < self.hparams.warmup_epochs:
            factor = (self.current_epoch + 1) / self.hparams.warmup_epochs
            for pg in self.optimizers().param_groups:
                pg["lr"] = self.hparams.lr * factor


# =============================================================================
# SYNTHETIC DATASET (for quick testing without real data)
# =============================================================================

class SyntheticOBBDataset(Dataset):
    """Synthetic dataset for pipeline testing without real data."""
    
    def __init__(self, num_samples: int = 100, img_size: int = 640, num_classes: int = 15):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = torch.rand(3, self.img_size, self.img_size)
        n = torch.randint(1, 10, (1,)).item()
        
        labels = torch.zeros(n, 6)
        labels[:, 0] = torch.randint(0, self.num_classes, (n,)).float()
        labels[:, 1:3] = torch.rand(n, 2) * 0.6 + 0.2  # x, y centered
        labels[:, 3:5] = torch.rand(n, 2) * 0.2 + 0.05  # w, h
        labels[:, 5] = torch.rand(n) * (math.pi / 2)  # angle
        
        return img, labels


# =============================================================================
# MAIN - Example usage
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLO-OBB with Lightning")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset YAML or root dir")
    parser.add_argument("--model", type=str, default="yolo11n-obb.pt", help="Model name or path")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    # Create model first so we know how many classes to expect in labels
    model = YOLOOBBLightning(model_name=args.model, lr=args.lr)
    num_classes = model.nc

    # Create datamodule
    if args.synthetic or args.data is None:
        logger.info("Using synthetic dataset for testing...")
        train_ds = SyntheticOBBDataset(num_samples=200, img_size=args.img_size, num_classes=num_classes)
        val_ds = SyntheticOBBDataset(num_samples=50, img_size=args.img_size, num_classes=num_classes)
        datamodule = OBBDataModule(train_ds, val_ds, args.batch_size, num_workers=0)
    elif args.data.endswith(".yaml"):
        logger.info(f"Loading dataset from YAML: {args.data}")
        datamodule = OBBDataModule.from_yaml(
            args.data,
            args.img_size,
            args.batch_size,
            args.workers,
            num_classes=num_classes,
        )
    else:
        logger.info(f"Loading dataset from directory: {args.data}")
        train_ds = YOLOOBBDataset(args.data, "train", args.img_size, num_classes=num_classes)
        val_ds = YOLOOBBDataset(args.data, "val", args.img_size, num_classes=num_classes)
        datamodule = OBBDataModule(train_ds, val_ds, args.batch_size, args.workers)
    
    def _report_invalid(ds, split: str):
        invalid = getattr(ds, "_invalid_label_total", 0)
        if invalid:
            logger.warning(f"{split}: dropped {invalid} labels outside [0, {num_classes - 1}]")

    if not args.synthetic and args.data is not None:
        _report_invalid(datamodule.train_dataset, "train")
        if datamodule.val_dataset is not None:
            _report_invalid(datamodule.val_dataset, "val")

    # Callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=3),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=callbacks,
        log_every_n_steps=10,
    )
    
    # Train
    trainer.fit(model, datamodule=datamodule)
    
    logger.info("\n✅ Training complete!")
