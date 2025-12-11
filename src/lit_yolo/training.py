"""
Training function and configuration for YOLO-OBB and standard detection.
"""

import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from lit_yolo.data import DetDataModule, OBBDataModule
from lit_yolo.models import LitYOLODet, LitYOLOOBB

logger = logging.getLogger(__name__)


def train_obb(
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

    lightning_model = LitYOLOOBB(model, nc, lr, weight_decay, warmup_epochs, img_size)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Check if precision requires CUDA
    requires_cuda = precision in ("16-mixed", "bf16-mixed", "16", "bf16")
    prec = precision if torch.cuda.is_available() or not requires_cuda else "32"
    if prec != precision:
        logger.warning("CUDA unavailable, using 32-bit precision")

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        precision=prec,
        callbacks=[
            ModelCheckpoint(
                dirpath=output / "checkpoints",
                filename="yolo-obb-{epoch:03d}-{val_loss:.4f}",
                monitor="val/loss",
                mode="min",
                save_top_k=save_top_k,
                save_last=True,
            ),
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


def train_detect(
    data: str,
    model: str = "yolo11n.pt",
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
    """Train standard YOLO detection model with PyTorch Lightning.

    Args:
        data: Dataset root (must contain images/ and labels/).
        model: Pretrained YOLO model.
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
    dm = DetDataModule(data, img_size, batch_size, workers, num_classes)
    nc = dm.num_classes  # Triggers auto-detection if needed

    lightning_model = LitYOLODet(model, nc, lr, weight_decay, warmup_epochs, img_size)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Check if precision requires CUDA
    requires_cuda = precision in ("16-mixed", "bf16-mixed", "16", "bf16")
    prec = precision if torch.cuda.is_available() or not requires_cuda else "32"
    if prec != precision:
        logger.warning("CUDA unavailable, using 32-bit precision")

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        precision=prec,
        callbacks=[
            ModelCheckpoint(
                dirpath=output / "checkpoints",
                filename="yolo-det-{epoch:03d}-{val_loss:.4f}",
                monitor="val/loss",
                mode="min",
                save_top_k=save_top_k,
                save_last=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        default_root_dir=str(output),
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
        gradient_clip_val=gradient_clip_val,
    )

    logger.info(
        f"YOLO Detection Training\n{'=' * 40}\n"
        f"Data: {data} | Model: {model} | Classes: {nc}\n"
        f"Epochs: {epochs} | Batch: {batch_size} | LR: {lr} | Size: {img_size}"
    )

    trainer.fit(lightning_model, datamodule=dm)
    logger.info(f"Done! Checkpoints: {output / 'checkpoints'}")


def create_synthetic_dataset(
    output: str = "./synthetic_dataset",
    num_samples: int = 100,
    split_ratio: float = 0.8,
    img_size: int = 640,
    class_mode: str = "shape",
    num_objects: int = 3,
    min_size_ratio: float = 0.1,
    max_size_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    """Create a synthetic dataset with geometric shapes for testing.

    Args:
        output: Output directory for the dataset.
        num_samples: Total number of samples to generate.
        split_ratio: Ratio of training samples (e.g., 0.8 = 80% train, 20% val).
        img_size: Size of generated images (square).
        class_mode: Classification mode - "shape" or "color".
        num_objects: Number of objects to place in each image.
        min_size_ratio: Minimum object size as ratio of image size.
        max_size_ratio: Maximum object size as ratio of image size.
        seed: Random seed for reproducibility.
    """
    from lit_yolo.data import BaseYOLODataModule

    dataset_path = BaseYOLODataModule.create_synthetic_dataset(
        root=output,
        num_samples=num_samples,
        split_ratio=split_ratio,
        img_size=img_size,
        class_mode=class_mode,
        num_objects=num_objects,
        min_size_ratio=min_size_ratio,
        max_size_ratio=max_size_ratio,
        seed=seed,
    )

    logger.info(f"Synthetic dataset created successfully at: {dataset_path}")
