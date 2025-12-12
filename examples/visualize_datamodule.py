"""
Example script demonstrating datamodule visualization.

This script shows how to use the visualize_batch() method to create
grid visualizations of training/validation batches with annotations.

Usage:
    python examples/visualize_datamodule.py --data /path/to/dataset --output viz.jpg
    python examples/visualize_datamodule.py --synthetic --output synthetic_viz.jpg
"""

import argparse
import tempfile
from pathlib import Path

from lit_yolo.data import DetDataModule, OBBDataModule


def _process_dataset(args):
    """Process dataset and create visualization."""
    # Create datamodule
    print(f"Loading dataset from {args.data}...")
    if args.obb:
        print("Using OBB DataModule (oriented bounding boxes)")
        datamodule = OBBDataModule(
            data=args.data,
            img_size=args.img_size,
            batch_size=args.batch_size,
        )
    else:
        print("Using Det DataModule (axis-aligned bounding boxes)")
        datamodule = DetDataModule(
            data=args.data,
            img_size=args.img_size,
            batch_size=args.batch_size,
        )

    # Setup datamodule
    datamodule.setup("fit")
    print(f"Detected {datamodule.num_classes} classes")

    # Visualize batch
    print(f"Visualizing {args.split} batch {args.batch_idx}...")
    grid = datamodule.visualize_batch(
        split=args.split,
        output_path=args.output,
        batch_idx=args.batch_idx,
        class_names=args.class_names,
    )

    print(f"✓ Visualization saved to {args.output}")
    print(f"  Grid shape: {grid.shape}")
    print(f"  Batch size: {args.batch_size}")
    print("\nVisualization complete!")


def main():
    parser = argparse.ArgumentParser(description="Visualize datamodule batches")
    parser.add_argument(
        "--data",
        type=str,
        help="Path to dataset root directory (with images/ and labels/ subdirs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="batch_visualization.jpg",
        help="Output path for visualization image (default: batch_visualization.jpg)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for visualization (default: 8)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Image size (default: 640)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Dataset split to visualize (default: train)",
    )
    parser.add_argument(
        "--batch-idx",
        type=int,
        default=0,
        help="Index of batch to visualize (default: 0 for first batch)",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        nargs="+",
        help="Optional class names for labels (e.g., --class-names cat dog bird)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Create and use a synthetic dataset for demonstration",
    )
    parser.add_argument(
        "--obb",
        action="store_true",
        help="Use OBB datamodule (default: standard detection)",
    )

    args = parser.parse_args()

    # Create synthetic dataset if requested
    if args.synthetic:
        print("Creating synthetic dataset...")
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "synthetic"

            if args.obb:
                dataset_path = OBBDataModule.create_synthetic_dataset(
                    root=data_root,
                    num_samples=50,
                    split_ratio=0.8,
                    img_size=args.img_size,
                    class_mode="shape",
                )
            else:
                dataset_path = DetDataModule.create_synthetic_dataset(
                    root=data_root,
                    num_samples=50,
                    split_ratio=0.8,
                    img_size=args.img_size,
                    class_mode="shape",
                )

            args.data = str(dataset_path)
            args.class_names = ["square", "triangle", "circle"]
            print(f"Synthetic dataset created at {dataset_path}")

            # Process the dataset before temp directory is cleaned up
            _process_dataset(args)

    if not args.data and not args.synthetic:
        parser.error("--data is required unless --synthetic is used")

    # Process the dataset (either synthetic or provided)
    if not args.synthetic:
        _process_dataset(args)


if __name__ == "__main__":
    main()
