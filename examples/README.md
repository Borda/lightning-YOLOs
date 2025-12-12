# Examples

This directory contains example scripts demonstrating features of the lightning-YOLOs framework.

## Overlap Threshold Demo

**File:** `overlap_threshold_demo.py`

Demonstrates the overlap threshold feature in synthetic dataset generation. This feature prevents objects from significantly overlapping with each other or extending outside image boundaries.

**Run the demo:**

```bash
python examples/overlap_threshold_demo.py
```

**What it shows:**

- Creating datasets with different overlap threshold values (0.0, 0.1, 0.3, 0.5)
- How threshold affects object placement density
- Trade-offs between strict and relaxed placement

**Key parameters:**

- `overlap_threshold=0.0` - No overlap allowed (strictest)
- `overlap_threshold=0.1` - Minimal overlap (strict)
- `overlap_threshold=0.3` - Balanced (default, recommended)
- `overlap_threshold=0.5` - More lenient overlap

## Usage in Your Code

```python
from lit_yolo.data import BaseDataModule

# Create synthetic dataset with custom overlap threshold
dataset_path = BaseDataModule.create_synthetic_dataset(
    root="./my_dataset",
    num_samples=100,
    split_ratio=0.8,
    img_size=640,
    overlap_threshold=0.3,  # Prevent objects from overlapping too much
)
```

## CLI Usage

```bash
# Create dataset with custom overlap threshold
python -m lit_yolo create dataset \
    --output ./my_dataset \
    --num_samples 100 \
    --overlap_threshold 0.3
```
