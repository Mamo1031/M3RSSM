"""Tests of `objective.py`."""

import torch
from torch import Tensor

from models.objective import likelihood


def test__likelihood(observation_bld: Tensor) -> None:
    """Test the `likelihood()` function."""
    prediction = target = observation_bld
    loss = likelihood(prediction, target, event_ndims=3)
    assert loss.shape == ()


def test__likelihood_with_scale() -> None:
    """Test the `likelihood()` function with custom scale."""
    prediction = torch.rand(2, 4, 3, 64, 64)
    target = torch.rand(2, 4, 3, 64, 64)
    loss = likelihood(prediction, target, event_ndims=3, scale=0.5)
    assert loss.shape == ()
