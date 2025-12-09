# lit_yolo Package

YOLO-OBB Training with PyTorch Lightning - Refactored Package Structure

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

**Note:** The old `requirements.txt` is still available for reference, but dependencies are now managed through `pyproject.toml`.

## Usage

### Command Line Interface

Run the training using the CLI:

```bash
# Using python -m (works without installation)
python -m lit_yolo --data /path/to/dataset --model yolo11n-obb.pt

# Using the installed command (after pip install)
lit-yolo --data /path/to/dataset --model yolo11n-obb.pt

# With a config file
lit-yolo --config config.yaml

# With custom parameters
lit-yolo --data /path/to/dataset --model yolo11n-obb.pt --epochs 50 --batch_size 16
```

### Python API

You can also import and use the package in your Python code:

```python
from lit_yolo import train, YOLOOBBLightning, OBBDataModule

# Train using the function
train(
    data="/path/to/dataset",
    model="yolo11n-obb.pt",
    epochs=100,
    batch_size=8,
    lr=1e-3
)

# Or use the components directly
from pytorch_lightning import Trainer

dm = OBBDataModule(data="/path/to/dataset", batch_size=8)
model = YOLOOBBLightning(model_name="yolo11n-obb.pt", num_classes=15)

trainer = Trainer(max_epochs=100)
trainer.fit(model, datamodule=dm)
```

## Package Structure

```
src/lit_yolo/
├── __init__.py      # Package initialization and exports
├── __main__.py      # CLI entry point
├── data.py          # Dataset, DataModule, and data utilities
├── models.py        # YOLOOBBLightning model
└── training.py      # Training function
```

## Features

- ✅ Logging with structured format
- ✅ jsonargparse CLI for easy configuration
- ✅ TorchMetrics mAP calculation (train + val)
- ✅ Automatic Mixed Precision (AMP)
- ✅ Auto class detection from dataset
- ✅ PyTorch Lightning integration
- ✅ Modular package structure

## Migration from Original File

The original `yolo_obb_lightning.py` has been refactored into a proper Python package:

- **data.py**: Contains all dataset and data loading logic
- **models.py**: Contains the Lightning module
- **training.py**: Contains the training function
- **__main__.py**: Provides CLI access via `python -m lit_yolo`

All functionality remains the same, but now organized into logical modules.
