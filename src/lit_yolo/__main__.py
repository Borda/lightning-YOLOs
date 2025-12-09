"""
CLI entry point for lit_yolo package.

Usage:
    python -m lit_yolo --data /path/to/dataset --model yolo11n-obb.pt
    python -m lit_yolo --config config.yaml
"""

from __future__ import annotations

import logging
import sys

from .training import train

logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    try:
        from jsonargparse import CLI
    except ImportError:
        logger.error("Install jsonargparse: pip install 'jsonargparse[signatures]'")
        sys.exit(1)
    CLI(train, as_positional=False)


if __name__ == "__main__":
    main()
