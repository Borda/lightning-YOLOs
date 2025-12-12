# lightning-YOLOs

YOLO Training with PyTorch Lightning - supports both OBB (Oriented Bounding Box) and standard object detection

## Installation

```bash
# Install in editable mode
pip install -e .

# Or install from source
pip install .
```

## Quick Start

### Standard Object Detection

```bash
# Run standard detection training with CLI
python -m lit_yolo train detect --data /path/to/dataset --model yolo11n.pt

# Or use the installed command
lit-yolo train detect --data /path/to/dataset --model yolo11n.pt
```

### Oriented Bounding Box Detection (OBB)

```bash
# Run OBB training with CLI
python -m lit_yolo train obb --data /path/to/dataset --model yolo11n-obb.pt

# Or use the installed command
lit-yolo train obb --data /path/to/dataset --model yolo11n-obb.pt
```

## Dataset Format

### Standard Detection Format
For standard object detection, labels should be in YOLO format:
```
class x_center y_center width height
```
All values are normalized (0-1).

### OBB Format
For oriented bounding box detection, labels should contain 4 corner points:
```
class x1 y1 x2 y2 x3 y3 x4 y4
```

## Downloading Public Datasets

We provide example notebooks showing how to download and use datasets from popular platforms:

### From Roboflow Universe
See [notebooks/download-dataset-roboflow.ipynb](notebooks/download-dataset-roboflow.ipynb) for examples including:
- Hard Hat Detection (construction safety)
- Blood Cell Detection (medical imaging)
- Playing Cards Detection
- And more...

### From Kaggle
See [notebooks/download-dataset-kaggle.ipynb](notebooks/download-dataset-kaggle.ipynb) for examples including:
- Trash Detection (environmental monitoring)
- Safety Helmet Detection
- Pothole Detection (infrastructure)
- Face Mask Detection
- And more...

### Example: Download DOTA v1.5 dataset (OBB)

```bash
wget https://www.ultralytics.com/assets/DOTAv1.5.zip
rm -rf DOTAv1.5 sample_data
unzip -qq DOTAv1.5.zip
python -m py_tree DOTAv1.5 -d 1
```

For detailed usage instructions, see [src/README.md](src/README.md).

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0). See the [LICENSE](LICENSE) file for details.

**Important**: This project depends on [Ultralytics](https://github.com/ultralytics/ultralytics), which is licensed under AGPL-3.0. When using this software with Ultralytics, the combined work is subject to AGPL-3.0 terms. See [NOTICE](NOTICE) for more details.