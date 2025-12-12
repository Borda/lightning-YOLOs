from pathlib import Path
from typing import Literal

import matplotlib
from matplotlib import pyplot as plt

from lit_yolo import OBBDataModule
from lit_yolo.data.data_modules import logger


def show_dataset(
    data: str,
    output: str | None = None,
    split: Literal["train", "val"] = "train",
    batch_size: int = 8,
    batch_idx: int = 0,
    img_size: int = 640,
    num_workers: int = 4,
    num_classes: int | None = None,
) -> None:
    """Visualize a batch from the dataset with annotations.

    Creates a grid image showing all samples in the specified batch with drawn
    bounding boxes (oriented or axis-aligned) and class labels. Class names are
    automatically loaded from the dataset YAML file if available, otherwise class
    indices are used. If output path is not provided, displays the image in a matplotlib window.

    Args:
        data: Path to dataset root directory (with images/ and labels/ subdirs).
        output: Output path for visualization image. If None, shows in matplotlib window.
        split: Dataset split to visualize ("train" or "val").
        batch_size: Number of images in the batch to visualize.
        batch_idx: Index of batch to visualize (0 for first batch).
        img_size: Image size for loading.
        num_workers: Number of dataloader workers.
        num_classes: Number of classes (auto-detected if None).

    Examples:
        From command line::

            lit-yolo show dataset --data /path/to/dataset --output viz.jpg
            lit-yolo show dataset --data /path/to/dataset --split val --batch_size 4
            lit-yolo show dataset --data /path/to/dataset
    """
    # Use OBB datamodule as it supports both oriented and plain bounding boxes
    logger.info(f"Loading dataset from {data}...")
    datamodule = OBBDataModule(
        data=data,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        num_classes=num_classes,
    )

    # Setup datamodule
    datamodule.setup("fit")
    logger.info(f"Detected {datamodule.num_classes} classes")

    # Log class names if available
    if datamodule.class_names:
        logger.info(f"Loaded class names from dataset YAML: {datamodule.class_names}")

    # Visualize batch
    logger.info(f"Visualizing {split} batch {batch_idx}...")
    fig, axes = datamodule.visualize_batch(
        split=split,
        batch_idx=batch_idx,
    )

    if output is not None:
        # Save the figure
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info(f"✓ Visualization saved to {output}")
        plt.close(fig)
    else:
        # Check if we have a GUI backend available
        backend = matplotlib.get_backend()
        if backend.lower() in ("agg", "cairo", "pdf", "pgf", "ps", "svg", "template"):
            logger.warning(f"No GUI backend available (current: {backend}). Cannot display interactive window.")
            logger.info("Running in headless mode. Please specify --output to save to a file instead.")
            plt.close(fig)
            return

        # Display in matplotlib window
        logger.info("Displaying visualization in matplotlib window...")
        try:
            plt.show()
        except Exception as e:
            import warnings

            warnings.warn(
                f"Failed to display matplotlib window: {e}. "
                "Please specify --output to save to a file instead, or ensure GUI backend is available.",
                UserWarning,
                stacklevel=2,
            )
        finally:
            plt.close(fig)

    logger.info(f"  Batch size: {batch_size}")
    logger.info("Visualization complete!")
