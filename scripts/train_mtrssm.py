"""Training script for vision-only MTRSSM."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _train_common import run_training

if __name__ == "__main__":
    default_config = "src/models/mtrssm/configs/default.yaml"
    run_training(default_config)
