# M3RSSM

![python](https://img.shields.io/badge/python-3.12-blue)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/Mamo1031/M3RSSM/actions/workflows/ci.yaml/badge.svg)](https://github.com/Mamo1031/M3RSSM/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/Mamo1031/M3RSSM/graph/badge.svg?token=D2A2FU8CFY)](https://codecov.io/gh/Mamo1031/M3RSSM)

**M**ixture-of-Products-of-Experts **M**ultiple Timescale **M**ultimodal Recurrent State-Space Model (M3RSSM).

## Models

| Model | Modality | Description |
|-------|----------|-------------|
| **M3RSSM** | Vision + Left Tactile + Right Tactile + Action | A multi-timescale multimodal model combining MoPoE with MTRSSM. |
| **MoPoE-MRSSM** | Vision + Left Tactile + Right Tactile + Action | A multimodal model using MoPoE fusion. |
| **MTRSSM** | Vision + Action | A unimodal model with multiple timescales. |


## Setup

### Required Environment

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) (package manager)

### Installation

```bash
# Create virtual environment
uv venv

# Install dependencies (including development and training dependencies)
uv sync --editable --all-extras
```

`--all-extras` flag installs the following optional dependencies:
- **dev**: Development tools (mypy, ruff, pytest, etc.)
- **train**: Training libraries (torchvision, wandb, h5py, etc.)


## Usage

### Training

#### Available Models

| Model | Command | Config Path |
|-------|---------|-------------|
| M3RSSM | `train-m3rssm` | `src/models/m3rssm/configs/default.yaml` |
| MoPoE-MRSSM | `train-mopoe-mrssm` | `src/models/mopoe_mrssm/configs/default.yaml` |
| MTRSSM | `train-mtrssm` | `src/models/mtrssm/configs/default.yaml` |


#### Training Commands

**Use default configuration:**
```bash
uv run poe <command>
# Example: uv run poe train-m3rssm
```

**Specify a custom configuration file:**
```bash
uv run poe <command> -c <path_to_config.yaml>
# Example: uv run poe train-m3rssm -c src/models/m3rssm/configs/custom.yaml
```

**Run in background (with logging):**
```bash
nohup uv run poe <command> > <log_file>.log 2>&1 &
# Example: nohup uv run poe train-m3rssm > train_m3rssm.log 2>&1 &
```


### Configuration Files

Each model has a dedicated configuration file (YAML). See the table above for config paths.

The configuration file can be customized with the following items:

- **Model Architecture**: Encoder, decoder, representation network, transition network, MTRNN parameters (τ values)
- **Optimization**: Optimizer, learning rate scheduler configuration
- **Data**: Dataset, batch size, preprocessing configuration (vision, left/right tactile)
- **Training**: Number of epochs, precision, gradient clipping, etc.
- **Logging**: WandB project name, logging frequency, etc.

### Dataset

The data in the `data/` directory is automatically preprocessed. Supported formats:
- NumPy arrays (`.npy`)
- HDF5 files (`.h5`)


## Project Structure

```
M3RSSM/
├── src/models/
│   ├── core.py                    # Base RSSM class
│   ├── networks.py                # Network definitions (Encoder, Decoder, Representation, Transition)
│   ├── dataset.py                 # Base data loader
│   ├── state.py                   # State class for RSSM
│   ├── objective.py               # Loss functions
│   ├── callback.py                # Common callbacks (WandB logging, visualization)
│   ├── transform.py               # Data transforms (normalization, augmentation)
│   │
│   ├── mtrssm/                    # Vision-only MTRSSM
│   │   ├── core.py                # MTRSSM model
│   │   ├── dataset.py             # Single-modality data loader
│   │   ├── callback.py            # MTRSSM-specific callbacks
│   │   └── configs/
│   │       └── default.yaml       # Default configuration
│   │
│   ├── mopoe_mrssm/               # MoPoE-MRSSM (3-modality)
│   │   ├── core.py                # MoPoE-MRSSM model with PoE/MoE fusion
│   │   ├── dataset.py             # 3-modality data loader
│   │   ├── callback.py            # Multimodal visualization callbacks
│   │   └── configs/
│   │       └── default.yaml       # Default configuration
│   │
│   └── m3rssm/                    # M3RSSM (multimodal + multi-timescale)
│       ├── core.py                # M3RSSM model with MTRNN and MoPoE
│       ├── state.py               # MTState for hierarchical states
│       ├── callback.py            # M3RSSM-specific callbacks
│       └── configs/
│           └── default.yaml       # Default configuration
│
├── scripts/                       # Training scripts
│   ├── _train_common.py           # Common training utilities
│   ├── train_mtrssm.py            # MTRSSM training script
│   ├── train_mopoe_mrssm.py       # MoPoE-MRSSM training script
│   └── train_m3rssm.py            # M3RSSM training script
│
└── pyproject.toml                 # Project configuration
```


## Development

### Code quality check

This project uses the following tools:

- **Ruff**: Linter/formatter
- **mypy**: Type checking
- **pydoclint**: Documentation string check

#### Run code quality checks (with auto-fix)
```bash
uv run poe lint
```

### Run tests

```bash
uv run poe test
```
