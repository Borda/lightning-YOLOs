# Examples

This directory contains example scripts demonstrating how to use Lightning-YOLOs.

## Visualize DataModule

The `visualize_datamodule.py` script demonstrates how to visualize batches from your dataset with annotations.

### Basic Usage

```bash
# Visualize with synthetic dataset
python examples/visualize_datamodule.py --synthetic --output viz.jpg

# Visualize with your own dataset
python examples/visualize_datamodule.py --data /path/to/dataset --output viz.jpg

# Visualize validation set
python examples/visualize_datamodule.py --data /path/to/dataset --split val --output viz_val.jpg

# Use OBB (oriented bounding boxes)
python examples/visualize_datamodule.py --synthetic --obb --output viz_obb.jpg
```

### Options

- `--data`: Path to dataset root directory (must contain `images/` and `labels/` subdirectories)
- `--output`: Output path for visualization image (default: `batch_visualization.jpg`)
- `--batch-size`: Number of images in the batch to visualize (default: 8)
- `--img-size`: Image size (default: 640)
- `--split`: Dataset split to visualize - `train` or `val` (default: `train`)
- `--batch-idx`: Index of batch to visualize (default: 0 for first batch)
- `--class-names`: Optional class names for labels (e.g., `--class-names cat dog bird`)
- `--synthetic`: Create and use a synthetic dataset for demonstration
- `--obb`: Use OBB datamodule for oriented bounding boxes (default: standard detection)

### Examples

**Visualize first training batch with custom class names:**
```bash
python examples/visualize_datamodule.py \
    --data /path/to/dataset \
    --class-names person car truck \
    --batch-size 4 \
    --output train_batch_0.jpg
```

**Visualize second validation batch:**
```bash
python examples/visualize_datamodule.py \
    --data /path/to/dataset \
    --split val \
    --batch-idx 1 \
    --output val_batch_1.jpg
```

**Quick demo with synthetic dataset:**
```bash
python examples/visualize_datamodule.py --synthetic
```

## Using Visualization in Code

You can also use the visualization methods directly in your Python code:

```python
from lit_yolo.data import DetDataModule, OBBDataModule

# Create datamodule
dm = DetDataModule(data="/path/to/dataset", img_size=640, batch_size=8)
dm.setup("fit")

# Visualize training batch
grid = dm.visualize_batch(
    split="train",
    output_path="visualization.jpg",
    class_names=["cat", "dog", "bird"]
)

# Grid is returned as numpy array if you want to process it further
print(f"Grid shape: {grid.shape}")
```

For OBB (oriented bounding boxes):

```python
from lit_yolo.data import OBBDataModule

dm = OBBDataModule(data="/path/to/dataset", img_size=640, batch_size=8)
dm.setup("fit")

# Visualize with oriented boxes
grid = dm.visualize_batch(split="train", output_path="obb_viz.jpg")
```
