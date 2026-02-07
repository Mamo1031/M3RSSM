"""Tests of `callback.py`."""

from unittest.mock import MagicMock

import torch

from models.callback import WandBMetricOrganizer, denormalize_tensor


def test__denormalize_tensor() -> None:
    """Test denormalize_tensor from [-1, 1] to [0, 1]."""
    # Single value
    tensor = torch.tensor([-1.0, 0.0, 1.0])
    result = denormalize_tensor(tensor)
    expected = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(result, expected, atol=1e-6)

    # Batch of images
    tensor = torch.rand(2, 3, 64, 64) * 2 - 1
    result = denormalize_tensor(tensor)
    assert result.min() >= -1e-6
    assert result.max() <= 1.0 + 1e-6
    assert result.shape == tensor.shape


def test__denormalize_tensor_boundary() -> None:
    """Test denormalize_tensor at boundary values."""
    tensor = torch.tensor([[[-1.0, 1.0]]])
    result = denormalize_tensor(tensor)
    assert torch.allclose(result[0, 0, 0], torch.tensor(0.0))
    assert torch.allclose(result[0, 0, 1], torch.tensor(1.0))


def test__wandb_metric_organizer_init() -> None:
    """Test WandBMetricOrganizer initialization."""
    callback = WandBMetricOrganizer()
    assert callback.train_metrics == {"recon": [], "kl": [], "loss": []}
    assert callback.val_metrics == {"recon": [], "kl": [], "loss": []}


def test__wandb_metric_organizer_on_train_epoch_end() -> None:
    """Test WandBMetricOrganizer collects training metrics."""
    callback = WandBMetricOrganizer()
    trainer = MagicMock()
    trainer.current_epoch = 0
    trainer.logged_metrics = {
        "train/loss": torch.tensor(0.5),
        "train/recon": torch.tensor(0.3),
        "train/kl": torch.tensor(0.2),
    }

    callback.on_train_epoch_end(trainer, MagicMock())
    assert len(callback.train_metrics["loss"]) == 1
    assert callback.train_metrics["loss"][0] == (0, 0.5)


def test__wandb_metric_organizer_on_train_start_no_wandb() -> None:
    """Test WandBMetricOrganizer handles non-WandB logger."""
    callback = WandBMetricOrganizer()
    trainer = MagicMock()
    trainer.logger = None
    callback.on_train_start(trainer, MagicMock())
