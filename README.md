# lightning-YOLOs

YOLO-OBB Training with PyTorch Lightning

## Installation

```bash
# Install in editable mode
pip install -e .

# Or install from source
pip install .
```

## Quick Start

```bash
# Run training with CLI
python -m lit_yolo train --data /path/to/dataset --model yolo11n-obb.pt

# Or use the installed command
lit-yolo train --data /path/to/dataset --model yolo11n-obb.pt
```

## Example: Download DOTA v1.5 dataset

```bash
wget https://www.ultralytics.com/assets/DOTAv1.5.zip
rm -rf DOTAv1.5 sample_data
unzip -qq DOTAv1.5.zip
python -m py_tree DOTAv1.5 -d 1
```

For detailed usage instructions, see [src/README.md](src/README.md).