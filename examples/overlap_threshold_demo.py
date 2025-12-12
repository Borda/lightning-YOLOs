"""Demo script showing how overlap threshold affects synthetic dataset generation.

This script demonstrates the overlap threshold feature by generating samples with
different threshold values and comparing the results.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from lit_yolo.data import BaseDataModule


def demo_overlap_threshold():
    """Demonstrate overlap threshold functionality."""
    print("=" * 70)
    print("Overlap Threshold Demo for Synthetic Dataset Generation")
    print("=" * 70)

    # Create a temporary directory for the demo
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)

        # Demo 1: Low overlap threshold (strict placement)
        print("\n1. Creating dataset with LOW overlap threshold (0.1)")
        print("   Objects will have minimal overlap with each other and boundaries")
        dataset_low = BaseDataModule.create_synthetic_dataset(
            root=base_path / "low_overlap",
            num_samples=5,
            split_ratio=0.8,
            img_size=640,
            class_mode="shape",
            min_objects=5,
            max_objects=5,
            min_size_ratio=0.15,
            max_size_ratio=0.2,
            overlap_threshold=0.1,  # Low threshold - strict placement
            seed=42,
        )
        train_images_low = len(list((dataset_low / "images" / "train").glob("*.jpg")))
        print(f"   ✓ Created {train_images_low} training images at: {dataset_low}")

        # Demo 2: Default overlap threshold (balanced)
        print("\n2. Creating dataset with DEFAULT overlap threshold (0.3)")
        print("   Balanced between object density and overlap prevention")
        dataset_default = BaseDataModule.create_synthetic_dataset(
            root=base_path / "default_overlap",
            num_samples=5,
            split_ratio=0.8,
            img_size=640,
            class_mode="shape",
            min_objects=5,
            max_objects=5,
            min_size_ratio=0.15,
            max_size_ratio=0.2,
            overlap_threshold=0.3,  # Default threshold
            seed=42,
        )
        train_images_default = len(list((dataset_default / "images" / "train").glob("*.jpg")))
        print(f"   ✓ Created {train_images_default} training images at: {dataset_default}")

        # Demo 3: High overlap threshold (relaxed placement)
        print("\n3. Creating dataset with HIGH overlap threshold (0.5)")
        print("   More lenient - allows more objects to be placed with some overlap")
        dataset_high = BaseDataModule.create_synthetic_dataset(
            root=base_path / "high_overlap",
            num_samples=5,
            split_ratio=0.8,
            img_size=640,
            class_mode="shape",
            min_objects=5,
            max_objects=5,
            min_size_ratio=0.15,
            max_size_ratio=0.2,
            overlap_threshold=0.5,  # High threshold - relaxed placement
            seed=42,
        )
        train_images_high = len(list((dataset_high / "images" / "train").glob("*.jpg")))
        print(f"   ✓ Created {train_images_high} training images at: {dataset_high}")

        # Demo 4: Zero overlap threshold (no overlap allowed)
        print("\n4. Creating dataset with ZERO overlap threshold (0.0)")
        print("   Strictest setting - no overlap allowed at all")
        dataset_zero = BaseDataModule.create_synthetic_dataset(
            root=base_path / "zero_overlap",
            num_samples=5,
            split_ratio=0.8,
            img_size=640,
            class_mode="shape",
            min_objects=5,
            max_objects=5,
            min_size_ratio=0.1,
            max_size_ratio=0.15,
            overlap_threshold=0.0,  # Zero threshold - no overlap
            seed=42,
        )
        train_images_zero = len(list((dataset_zero / "images" / "train").glob("*.jpg")))
        print(f"   ✓ Created {train_images_zero} training images at: {dataset_zero}")

        # Summary
        print("\n" + "=" * 70)
        print("Summary:")
        print("=" * 70)
        print(f"  Low threshold (0.1):     {train_images_low} images created")
        print(f"  Default threshold (0.3): {train_images_default} images created")
        print(f"  High threshold (0.5):    {train_images_high} images created")
        print(f"  Zero threshold (0.0):    {train_images_zero} images created")
        print("\nKey Insights:")
        print("  - Lower thresholds may result in fewer objects per image")
        print("  - Higher thresholds allow denser object packing")
        print("  - Zero threshold is strictest but may struggle with large objects")
        print("  - Default (0.3) provides good balance for most use cases")
        print("\nUsage in CLI:")
        print("  python -m lit_yolo create dataset --overlap_threshold 0.3")
        print("\nUsage in Python:")
        print("  BaseDataModule.create_synthetic_dataset(..., overlap_threshold=0.3)")


if __name__ == "__main__":
    demo_overlap_threshold()
