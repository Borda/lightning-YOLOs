# lightning-YOLOs

YOLO Training with PyTorch Lightning - supports both OBB (Oriented Bounding Box) and standard object detection

Supports:
- **Standard Object Detection**: Axis-aligned bounding boxes for general object detection
- **Oriented Bounding Box (OBB) Detection**: Rotated bounding boxes for aerial/satellite imagery

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

## Installation

```bash
# Install in editable mode (recommended for development)
pip install -e .

# Or install from source
pip install .

# Or install with optional development dependencies
pip install -e ".[dev]"
```

## Quick Start

### Command Line Interface

#### Standard Object Detection

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

## Creating a Synthetic Dataset

For testing and validation, you can create a synthetic dataset with basic geometric shapes:

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
    --min_objects 3 --max_objects 3 \
    --seed 42

# Create with color-based classification
lit-yolo create dataset \
    --output ./synthetic_colors \
    --class_mode color \
    --num_samples 100

# Then use the synthetic dataset for training
lit-yolo train detect --data ./synthetic_dataset --model yolo11n.pt --epochs 10
```

For details on all available parameters, run:
```bash
lit-yolo create dataset --help
```

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

### Standard Detection Format

Labels should be in standard YOLO format (one file per image in `labels/` directory):

```
class_id x_center y_center width height
```

All values are normalized between 0 and 1. Example:
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.15 0.2
```

### OBB Format

Labels should contain 4 corner points (one file per image in `labels/` directory):

```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

All values are normalized between 0 and 1.


## Dataset Format Support

The OBB dataset loader supports both formats:

1. **OBB Format** (9 values per line): `class x1 y1 x2 y2 x3 y3 x4 y4`
   - Recommended for production training with oriented bounding boxes
   - Provides full rotation information

2. **Standard Detection Format** (5 values per line): `class x y w h`
   - Useful for debugging convergence with simpler datasets
   - Enables CI testing with tiny datasets
   - Rotation is automatically set to 0
   - A warning is logged when this format is detected

Both formats can coexist in the same project (e.g., different splits can use different formats).

## Downloading Public Datasets

### From Roboflow Universe

Roboflow provides thousands of public datasets for computer vision. Here's how to download and use them:

```bash
# Install roboflow package
pip install roboflow
```

```python
from roboflow import Roboflow

# Initialize with your API key (get free key from https://app.roboflow.com)
rf = Roboflow(api_key="YOUR_API_KEY")

# Example: Hard Hat Detection dataset (construction safety, 3 classes)
project = rf.workspace("roboflow-universe").project("hard-hat-workers")
dataset = project.version(1).download("yolov11")

# Train with Lightning-YOLOs
from lit_yolo import train_detect
train_detect(data=f"{dataset.location}/data.yaml", model="yolo11n.pt", epochs=50)
```

**Popular datasets on Roboflow:**
- Hard Hat Detection (construction safety, 3 classes, ~7K images)
- Blood Cell Detection (medical imaging, 3 classes, ~360 images)
- Playing Cards Detection (53 classes, ~7.6K images)

Browse more at [Roboflow Universe](https://universe.roboflow.com/)

### From Kaggle

Kaggle hosts many object detection datasets. Here's how to download them:

```bash
# Install kaggle package
pip install kaggle

# Setup: Download kaggle.json from https://www.kaggle.com/settings/account
# Place it in ~/.kaggle/kaggle.json (Linux/Mac) or C:\Users\<username>\.kaggle\kaggle.json (Windows)
# Set permissions: chmod 600 ~/.kaggle/kaggle.json
```

```bash
# Example: Hard Hat Detection dataset
kaggle datasets download -d andrewmvd/hard-hat-detection
unzip -q hard-hat-detection.zip -d helmet_dataset

# Create data.yaml file (adjust paths and class names for your dataset)
cat > helmet_data.yaml << EOF
train: helmet_dataset/images/train
val: helmet_dataset/images/valid
test: helmet_dataset/images/test  # optional

nc: 3  # number of classes
names: ["head", "helmet", "person"]  # adjust class names as needed
EOF

# Train with Lightning-YOLOs
lit-yolo train detect --data helmet_data.yaml --model yolo11n.pt --epochs 50
```

**Popular datasets on Kaggle:**
- Trash Detection / TACO (environmental monitoring, 60+ classes, ~1.5K images)
- Safety Helmet Detection (construction safety, 3 classes, ~5K images)
- Pothole Detection (infrastructure monitoring, 2 classes, ~665 images)
- Face Mask Detection (public health, 3 classes, ~12K images)

Browse more at [Kaggle Datasets](https://www.kaggle.com/datasets)

### Example: Download DOTA v1.5 dataset (OBB)

```bash
wget https://www.ultralytics.com/assets/DOTAv1.5.zip
rm -rf DOTAv1.5 sample_data
unzip -qq DOTAv1.5.zip
python -m py_tree DOTAv1.5 -d 1
```


## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0). See the [LICENSE](LICENSE) file for details.

**Important**: This project depends on [Ultralytics](https://github.com/ultralytics/ultralytics), which is licensed under AGPL-3.0. When using this software with Ultralytics, the combined work is subject to AGPL-3.0 terms. See [NOTICE](NOTICE) for more details.