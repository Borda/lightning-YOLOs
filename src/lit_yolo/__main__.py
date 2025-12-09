"""
CLI entry point for lit_yolo package.
"""

import logging
import sys

# Configure logging when running as CLI
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from .training import train

logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    try:
        from jsonargparse import CLI
    except ImportError:
        logger.error("Install jsonargparse: pip install 'jsonargparse[signatures]'")
        sys.exit(1)
    
    # Use dictionary-based CLI for subcommands
    CLI({"train": train}, as_positional=False)


if __name__ == "__main__":
    main()
