"""Common training script utilities."""

import sys
from pathlib import Path

from lightning.pytorch.cli import LightningCLI


def run_training(default_config: str | Path) -> None:
    """Run training with the specified default config.

    Args:
        default_config: Path to the default config file.
    """
    default_config_str = str(default_config)
    has_config = any(arg in {"-c", "--config"} for arg in sys.argv[1:])

    if not has_config:
        sys.argv.insert(1, "fit")
        sys.argv.insert(2, "-c")
        sys.argv.insert(3, default_config_str)
    elif "fit" not in sys.argv[1:]:
        sys.argv.insert(1, "fit")

    LightningCLI(
        run=True,
        save_config_callback=None,
    )
