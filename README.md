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

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0). See the [LICENSE](LICENSE) file for details.

**Important**: This project depends on [Ultralytics](https://github.com/ultralytics/ultralytics), which is licensed under AGPL-3.0. When using this software with Ultralytics, the combined work is subject to AGPL-3.0 terms. See [NOTICE](NOTICE) for more details.