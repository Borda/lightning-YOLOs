"""
CLI entry point for lit_yolo package.

Usage:
    python -m lit_yolo train obb --data /path/to/dataset --model yolo11n-obb.pt
    python -m lit_yolo train detect --data /path/to/dataset --model yolo11n.pt
    python -m lit_yolo train obb --config config.yaml
    python -m lit_yolo show dataset --data /path/to/dataset --output viz.jpg
    python -m lit_yolo create dataset --output ./synthetic_dataset
"""

import logging
import sys

# Configure logging when running as CLI
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from lit_yolo.data import create_synthetic_dataset
from lit_yolo.data.data_modules import show_dataset
from lit_yolo.training import train_detect, train_obb

logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    try:
        from jsonargparse import CLI
    except ImportError:
        logger.error("Install jsonargparse: pip install 'jsonargparse[signatures]'")
        sys.exit(1)

    # Use dictionary-based CLI for subcommands
    CLI(
        {
            "train": {"obb": train_obb, "detect": train_detect},
            "create": {"dataset": create_synthetic_dataset},
            "show": {"dataset": show_dataset},
        },
        as_positional=False,
    )


if __name__ == "__main__":
    main()
