"""Training script for M3RSSM."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _train_common import run_training

if __name__ == "__main__":
    default_config = "src/models/m3rssm/configs/default.yaml"
    run_training(default_config)
