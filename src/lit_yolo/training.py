"""
Training function and configuration for YOLO-OBB.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from .data import OBBDataModule
from .models import YOLOOBBLightning

logger = logging.getLogger(__name__)


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
