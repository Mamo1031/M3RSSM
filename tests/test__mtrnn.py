"""Tests of MTRNN cell (from mtrssm/core.py and m3rssm/core.py)."""

import pytest
import torch

from models.mtrssm.core import MTRNN


def test__mtrnn_forward() -> None:
    """Test MTRNN forward pass."""
    batch_size = 4
    input_dim = 16
    hidden_dim = 8
    mtrnn = MTRNN(input_dim=input_dim, hidden_dim=hidden_dim, tau=4.0)

    inputs = torch.rand(batch_size, input_dim)
    prev_d = torch.rand(batch_size, hidden_dim)

    output = mtrnn(inputs, prev_d)
    assert output.shape == (batch_size, hidden_dim)
    assert output.min() >= -1.0
    assert output.max() <= 1.0


def test__mtrnn_sequential_calls() -> None:
    """Test MTRNN with sequential calls (hidden state persistence)."""
    mtrnn = MTRNN(input_dim=8, hidden_dim=4, tau=2.0)
    inputs = torch.rand(2, 8)
    prev_d = torch.rand(2, 4)

    out1 = mtrnn(inputs, prev_d)
    out2 = mtrnn(inputs, prev_d)
    assert out1.shape == out2.shape
    assert mtrnn.hidden is not None


def test__mtrnn_invalid_tau() -> None:
    """Test MTRNN raises ValueError for tau <= 1.0."""
    with pytest.raises(ValueError, match=r"tau must be greater than 1\.0"):
        MTRNN(input_dim=8, hidden_dim=4, tau=1.0)
    with pytest.raises(ValueError, match=r"tau must be greater than 1\.0"):
        MTRNN(input_dim=8, hidden_dim=4, tau=0.5)
