# lit_yolo Package

YOLO Training with PyTorch Lightning - Refactored Package Structure

Supports both:
- **Standard Object Detection**: Axis-aligned bounding boxes for general object detection
- **Oriented Bounding Box (OBB) Detection**: Rotated bounding boxes for aerial/satellite imagery

## Installation

The package can be installed using pip with the included `pyproject.toml`:

```bash
# Install in editable mode (recommended for development)
pip install -e .

# Or install from the repository root
pip install .

# Or install with optional development dependencies
pip install -e ".[dev]"
```

## Usage

### Command Line Interface

#### Standard Object Detection

Run standard detection training using the CLI:

```bash
# Using python -m (works without installation)
python -m lit_yolo train detect --data /path/to/dataset --model yolo11n.pt

# Using the installed command (after pip install)
lit-yolo train detect --data /path/to/dataset --model yolo11n.pt

# With a config file
lit-yolo train detect --config config.yaml

# With custom parameters
lit-yolo train detect --data /path/to/dataset --model yolo11n.pt --epochs 50 --batch_size 16
```

#### Oriented Bounding Box (OBB) Detection

Run OBB training using the CLI:

```bash
# Using python -m (works without installation)
python -m lit_yolo train obb --data /path/to/dataset --model yolo11n-obb.pt

# Using the installed command (after pip install)
lit-yolo train obb --data /path/to/dataset --model yolo11n-obb.pt

# With a config file
lit-yolo train obb --config config.yaml

# With custom parameters
lit-yolo train obb --data /path/to/dataset --model yolo11n-obb.pt --epochs 50 --batch_size 16
```

### Python API

You can also import and use the package in your Python code:

#### Standard Object Detection

```python
from lit_yolo import train_detect, LitYOLODet, DetDataModule

# Train using the function
train_detect(
    data="/path/to/dataset",
    model="yolo11n.pt",
    epochs=100,
    batch_size=8,
    lr=1e-3
)

# Or use the components directly
from pytorch_lightning import Trainer

dm = DetDataModule(data="/path/to/dataset", batch_size=8)
model = LitYOLODet(model_name="yolo11n.pt", num_classes=80)

trainer = Trainer(max_epochs=100)
trainer.fit(model, datamodule=dm)
```

#### Oriented Bounding Box (OBB) Detection

```python
from lit_yolo import train_obb, LitYOLOOBB, OBBDataModule

# Train using the function
train_obb(
    data="/path/to/dataset",
    model="yolo11n-obb.pt",
    epochs=100,
    batch_size=8,
    lr=1e-3
)

# Or use the components directly
from pytorch_lightning import Trainer

dm = OBBDataModule(data="/path/to/dataset", batch_size=8)
model = LitYOLOOBB(model_name="yolo11n-obb.pt", num_classes=15)

trainer = Trainer(max_epochs=100)
trainer.fit(model, datamodule=dm)
```

### Creating a Synthetic Dataset

For testing and validation, you can create a synthetic dataset with basic geometric shapes using the CLI:

```bash
# Create synthetic dataset with default settings (shape-based classification)
lit-yolo create dataset --output ./synthetic_dataset

# Create with custom settings
lit-yolo create dataset \
    --output ./synthetic_shapes \
    --num_samples 150 \
    --split_ratio 0.8 \
    --img_size 640 \
    --class_mode shape \
    --num_objects 3 \
    --seed 42

# Create with color-based classification
lit-yolo create dataset \
    --output ./synthetic_colors \
    --class_mode color \
    --num_samples 100

# Then use the synthetic dataset for training
lit-yolo train detect --data ./synthetic_dataset --model yolo11n.pt --epochs 10
```

**Parameters:**
- `--output`: Output directory for the dataset (default: `./synthetic_dataset`)
- `--num_samples`: Total number of samples to generate (default: `100`)
- `--split_ratio`: Train/val split ratio, e.g., 0.8 = 80% train, 20% val (default: `0.8`)
- `--img_size`: Size of generated square images (default: `640`)
- `--class_mode`: Classification mode - `shape` (3 shape classes) or `color` (3 color classes) (default: `shape`)
- `--num_objects`: Number of objects per image (default: `3`)
- `--min_size_ratio`: Minimum object size as ratio of image size (default: `0.1`)
- `--max_size_ratio`: Maximum object size as ratio of image size (default: `0.2`)
- `--seed`: Random seed for reproducibility (default: `42`)

The synthetic dataset feature generates:
- Images with three basic shapes: square, triangle, and circle
- Each shape rendered in a different color: red, green, and blue
- Valid YOLO format labels for all objects
- Both training and validation splits

This is useful for:
- Testing the implementation is working correctly
- Verifying that metrics and loss are converging
- Quick validation of changes without downloading large datasets

## Package Structure

```
src/lit_yolo/
├── __init__.py      # Package initialization and exports
├── __main__.py      # CLI entry point
├── data.py          # Dataset, DataModule, and data utilities
│                      - YOLODetDataset, DetDataModule (standard detection)
│                      - YOLOOBBDataset, OBBDataModule (OBB detection)
├── models.py        # Lightning modules
│                      - LitYOLODet (standard detection)
│                      - LitYOLOOBB (OBB detection)
└── training.py      # Training functions
                       - train_detect (standard detection)
                       - train_obb (OBB detection)
```

## Dataset Format

### Standard Object Detection

Labels should be in standard YOLO format (one file per image in `labels/` directory):

```
class_id x_center y_center width height
```

All values are normalized between 0 and 1. Example:
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.15 0.2
```

### Oriented Bounding Box (OBB)

Labels should contain 4 corner points (one file per image in `labels/` directory):

```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

All values are normalized between 0 and 1.

## Features

- ✅ Logging with structured format
- ✅ jsonargparse CLI for easy configuration
- ✅ TorchMetrics mAP calculation (train + val)
- ✅ Automatic Mixed Precision (AMP)
- ✅ Auto class detection from dataset
- ✅ PyTorch Lightning integration
- ✅ Modular package structure
- ✅ Standard detection and OBB support
- ✅ Synthetic dataset generation for testing and validation

### Dataset Format Support

The OBB dataset loader now supports both formats:

1. **OBB Format** (9 values per line): `class x1 y1 x2 y2 x3 y3 x4 y4`
   - Recommended for production training with oriented bounding boxes
   - Provides full rotation information

2. **Standard Detection Format** (5 values per line): `class x y w h`
   - Useful for debugging convergence with simpler datasets
   - Enables CI testing with tiny datasets
   - Rotation is automatically set to 0
   - A warning is logged when this format is detected

Both formats can coexist in the same project (e.g., different splits can use different formats).

## Migration from Original File

The original `yolo_obb_lightning.py` has been refactored and extended:

- **data.py**: Contains all dataset and data loading logic for both standard and OBB detection
- **models.py**: Contains the Lightning modules for both detection types
- **training.py**: Contains the training functions for both detection types
- **__main__.py**: Provides CLI access via `python -m lit_yolo`

All functionality remains the same, with the addition of standard object detection support.
