"""
YOLO-OBB Lightning module for training.
"""

import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytorch_lightning as pl
import torch
import yaml

from lit_yolo.data import obb_to_xyxy, xywh_to_xyxy

logger = logging.getLogger(__name__)


class BaseLitYOLO(pl.LightningModule):
    """
    Base LightningModule for YOLO training with common functionality.

    This class provides a foundation for training YOLO models using PyTorch Lightning.
    It handles model initialization, optimizer and scheduler configuration, training and validation
    step logic, and metric logging (e.g., mAP). It is designed to be subclassed for specific YOLO
    variants or custom training logic.

    Subclasses must implement:
        - `_update_metrics(self, preds: Any, batch: dict, metric)`: Update the provided metric object
          with predictions and ground truth from the batch. This is required for correct metric logging.

    Common functionality provided:
        - Model loading and initialization from a given model name or path.
        - Training and validation step logic, including loss computation and metric updates.
        - Optimizer and learning rate scheduler configuration.
        - Metric initialization in setup() and automatic logging for mAP (if torchmetrics[detection] is installed).
        - Device management and batch transfer utilities.

    Example:
        class MyYOLO(BaseLitYOLO):
            def _update_metrics(self, preds, batch, metric):
                # Custom metric update logic
                ...

    See `LitYOLOOBB` for a concrete implementation.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        warmup_epochs: int = 3,
        img_size: int = 640,
        default_nc: int = 80,
    ):
        """Initialize base YOLO Lightning module.

        Args:
            model_name: Name/path of the YOLO model to load.
            num_classes: Number of classes for the task.
            lr: Learning rate.
            weight_decay: L2 regularization weight.
            warmup_epochs: Number of warmup epochs.
            img_size: Input image size.
            default_nc: Default number of classes if not found in model config.
        """
        super().__init__()
        self.save_hyperparameters()

        from ultralytics import YOLO

        yolo = YOLO(model_name)
        model_nc = yolo.model.yaml.get("nc", num_classes)

        if num_classes != model_nc:
            cfg = {**yolo.model.yaml, "nc": num_classes}
            if "scale" not in cfg:
                cfg["scale"] = next(
                    (s for s in "nsmlx" if f"yolo11{s}" in model_name or f"yolov8{s}" in model_name), "n"
                )

            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                yaml.dump(cfg, f)
                temp_yaml = f.name
            try:
                yolo = YOLO(temp_yaml)
            finally:
                Path(temp_yaml).unlink(missing_ok=True)
            logger.info(f"Rebuilt model: {model_nc} -> {num_classes} classes")

        self.model = yolo.model
        self.nc = num_classes

        if not hasattr(self.model, "args") or self.model.args is None:
            self.model.args = {}
        if isinstance(self.model.args, dict):
            self.model.args = SimpleNamespace(**{**{"box": 7.5, "cls": 0.5, "dfl": 1.5}, **self.model.args})

        self.criterion = self.model.init_criterion()

        # Metrics will be initialized in setup() after device assignment
        self.train_map = None
        self.val_map = None

        logger.info(f"Model: {model_name} ({self.nc} classes)")

    def setup(self, stage: str | None = None):
        """Setup the model and metrics on the correct device."""
        for p in self.model.parameters():
            p.requires_grad = True
        self.criterion.device = self.device
        if hasattr(self.criterion, "proj"):
            self.criterion.proj = self.criterion.proj.to(self.device)

        # Initialize metrics on the correct device
        if self.train_map is None:
            try:
                from torchmetrics.detection import MeanAveragePrecision

                self.train_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox").to(self.device)
                self.val_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox").to(self.device)
                logger.info("Metrics enabled: train and val mAP")
            except ImportError:
                logger.warning("torchmetrics[detection] not installed, metrics disabled")

    def forward(self, x: torch.Tensor):
        """Forward pass through the model."""
        return self.model(x)

    def _to_device(self, batch: dict) -> dict:
        """Move batch tensors to the model's device."""
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def _compute_step(self, batch: dict, stage: str) -> torch.Tensor:
        """Shared logic for training and validation steps.

        Computes the loss, logs metrics, and updates the metric tracker by calling
        the abstract _update_metrics method (which must be implemented by subclasses).

        Args:
            batch: Dictionary containing 'img' and other batch data.
            stage: Either 'train' or 'val' to distinguish training/validation.

        Returns:
            Total loss as a scalar tensor.
        """
        preds = self(batch["img"])
        batch_dev = self._to_device(batch)
        loss, items = self.criterion(preds, batch_dev)
        total = loss.sum()

        self.log(
            f"{stage}/loss", total, prog_bar=True, on_step=(stage == "train"), on_epoch=True, sync_dist=(stage == "val")
        )
        self.log_dict(
            {f"{stage}/box": items[0], f"{stage}/cls": items[1], f"{stage}/dfl": items[2]},
            on_epoch=True,
            sync_dist=(stage == "val"),
        )

        # Update metrics
        metric = self.train_map if stage == "train" else self.val_map
        if metric is not None:
            try:
                self._update_metrics(preds, batch_dev, metric)
            except Exception as e:
                logger.warning(f"{stage} metrics failed: {e}")

        return total

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step."""
        return self._compute_step(batch, "train")

    @torch.no_grad()
    def validation_step(self, batch: dict, batch_idx: int):
        """Validation step."""
        return self._compute_step(batch, "val")

    def _update_metrics(self, preds: Any, batch: dict, metric):
        """Update metrics with predictions. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _update_metrics")

    def _log_metrics(self, stage: str):
        """Log computed metrics."""
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
        """Called at the end of training epoch."""
        self._log_metrics("train")

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        self._log_metrics("val")

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        decay = [p for n, p in self.model.named_parameters() if p.requires_grad and "bn" not in n and "bias" not in n]
        no_decay = [p for n, p in self.model.named_parameters() if p.requires_grad and ("bn" in n or "bias" in n)]

        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.hparams.lr,
        )

        # Use trainer max_epochs if available, otherwise fall back to a default
        max_epochs = getattr(getattr(self, "trainer", None), "max_epochs", 100)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, max_epochs - self.hparams.warmup_epochs), eta_min=self.hparams.lr * 0.01
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def on_train_epoch_start(self):
        """Handle learning rate warmup."""
        if self.current_epoch < self.hparams.warmup_epochs:
            factor = (self.current_epoch + 1) / self.hparams.warmup_epochs
            for pg in self.optimizers().param_groups:
                pg["lr"] = self.hparams.lr * factor


class LitYOLOOBB(BaseLitYOLO):
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
        """Initialize YOLO-OBB Lightning module.

        Args:
            model_name: Name/path of the YOLO-OBB model.
            num_classes: Number of classes.
            lr: Learning rate.
            weight_decay: L2 regularization weight.
            warmup_epochs: Number of warmup epochs.
            img_size: Input image size.
        """
        super().__init__(model_name, num_classes, lr, weight_decay, warmup_epochs, img_size, default_nc=15)

    def _update_metrics(self, preds: Any, batch: dict, metric):
        """Update metrics with OBB predictions."""
        from ultralytics.utils.nms import non_max_suppression

        raw = preds[0] if isinstance(preds, (list, tuple)) else preds
        nms_preds = non_max_suppression(raw, conf_thres=0.001, iou_thres=0.7, nc=self.nc, max_det=300, rotated=True)

        preds_list, targets_list = [], []
        img_size = self.hparams.img_size

        for i, pred in enumerate(nms_preds):
            mask = batch["batch_idx"] == i
            gt_cls = batch["cls"][mask].squeeze(-1) if mask.sum() else torch.empty(0, device=self.device)
            gt_box = batch["bboxes"][mask] if mask.sum() else torch.empty((0, 5), device=self.device)

            targets_list.append(
                {
                    "boxes": obb_to_xyxy(gt_box, img_size) if len(gt_box) else torch.empty((0, 4), device=self.device),
                    "labels": gt_cls.long(),
                }
            )

            has_pred = pred is not None and len(pred)
            preds_list.append(
                {
                    "boxes": obb_to_xyxy(pred[:, :5], img_size)
                    if has_pred
                    else torch.empty((0, 4), device=self.device),
                    "scores": pred[:, 5] if has_pred else torch.empty(0, device=self.device),
                    "labels": pred[:, 6].long() if has_pred else torch.empty(0, dtype=torch.long, device=self.device),
                }
            )

        metric.update(preds_list, targets_list)


class LitYOLODet(BaseLitYOLO):
    """Lightning module for standard YOLO detection training with TorchMetrics."""

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        num_classes: int = 80,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        warmup_epochs: int = 3,
        img_size: int = 640,
    ):
        """Initialize standard YOLO detection Lightning module.

        Args:
            model_name: Name/path of the YOLO model.
            num_classes: Number of classes.
            lr: Learning rate.
            weight_decay: L2 regularization weight.
            warmup_epochs: Number of warmup epochs.
            img_size: Input image size.
        """
        super().__init__(model_name, num_classes, lr, weight_decay, warmup_epochs, img_size, default_nc=80)

    def _update_metrics(self, preds: Any, batch: dict, metric):
        """Update metrics with standard detection predictions."""
        from ultralytics.utils.nms import non_max_suppression

        raw = preds[0] if isinstance(preds, (list, tuple)) else preds
        nms_preds = non_max_suppression(raw, conf_thres=0.001, iou_thres=0.7, nc=self.nc, max_det=300, rotated=False)

        preds_list, targets_list = [], []
        img_size = self.hparams.img_size

        for i, pred in enumerate(nms_preds):
            mask = batch["batch_idx"] == i
            gt_cls = batch["cls"][mask].squeeze(-1) if mask.sum() else torch.empty(0, device=self.device)
            gt_box = batch["bboxes"][mask] if mask.sum() else torch.empty((0, 4), device=self.device)

            targets_list.append(
                {
                    "boxes": xywh_to_xyxy(gt_box, img_size) if len(gt_box) else torch.empty((0, 4), device=self.device),
                    "labels": gt_cls.long(),
                }
            )

            has_pred = pred is not None and len(pred)
            preds_list.append(
                {
                    "boxes": xywh_to_xyxy(pred[:, :4], img_size)
                    if has_pred
                    else torch.empty((0, 4), device=self.device),
                    "scores": pred[:, 4] if has_pred else torch.empty(0, device=self.device),
                    "labels": pred[:, 5].long() if has_pred else torch.empty(0, dtype=torch.long, device=self.device),
                }
            )

        metric.update(preds_list, targets_list)
