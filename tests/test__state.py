"""Tests for `state.py`."""

import pytest
import torch
from distribution_extension import MultiOneHot

from models.state import State, cat_states, stack_states
from tests.conftest import (
    BATCH_SIZE,
    CATEGORY_SIZE,
    CLASS_SIZE,
    DETERMINISTIC_SIZE,
    SEQ_LEN,
)


@pytest.fixture
def state() -> State:
    """Create a State instance."""
    deter = torch.rand(
        BATCH_SIZE,
        SEQ_LEN,
        DETERMINISTIC_SIZE,
        requires_grad=True,
    )
    logit = torch.rand(BATCH_SIZE, SEQ_LEN, CATEGORY_SIZE, CLASS_SIZE, requires_grad=True)
    distribution = MultiOneHot(logit)
    return State(deter=deter, distribution=distribution)


def test_init(state: State) -> None:
    """Test the __init__ method."""
    assert state.deter.shape == (BATCH_SIZE, SEQ_LEN, DETERMINISTIC_SIZE)
    assert state.stoch.shape == (BATCH_SIZE, SEQ_LEN, CATEGORY_SIZE * CLASS_SIZE)
    assert state.feature.shape == (
        BATCH_SIZE,
        SEQ_LEN,
        DETERMINISTIC_SIZE + CATEGORY_SIZE * CLASS_SIZE,
    )


def test__iter__(state: State) -> None:
    """Test the __iter__ method."""
    for s in state:
        assert s.deter.shape == (SEQ_LEN, DETERMINISTIC_SIZE)
        assert s.stoch.shape == (SEQ_LEN, CATEGORY_SIZE * CLASS_SIZE)
        assert s.feature.shape == (
            SEQ_LEN,
            DETERMINISTIC_SIZE + CATEGORY_SIZE * CLASS_SIZE,
        )


def test__getitem__(state: State) -> None:
    """Test the __getitem__ method."""
    for t in range(SEQ_LEN):
        s = state[:, t]
        assert s.deter.shape == (BATCH_SIZE, DETERMINISTIC_SIZE)
        assert s.stoch.shape == (BATCH_SIZE, CATEGORY_SIZE * CLASS_SIZE)
        assert s.feature.shape == (
            BATCH_SIZE,
            DETERMINISTIC_SIZE + CATEGORY_SIZE * CLASS_SIZE,
        )


def test__to(state: State) -> None:
    """Test the to method."""
    state_moved = state.to(torch.device("cpu"))
    assert state_moved.deter.device == torch.device("cpu")
    assert state_moved.stoch.device == torch.device("cpu")


def test__detach(state: State) -> None:
    """Test the detach method."""
    assert state.deter.requires_grad is True
    assert state.stoch.requires_grad is True
    state_detached = state.detach()
    assert state_detached.deter.requires_grad is False
    assert state_detached.stoch.requires_grad is False


def test__clone(state: State) -> None:
    """Test the clone method."""
    clone = state.clone()
    assert clone.deter.shape == (BATCH_SIZE, SEQ_LEN, DETERMINISTIC_SIZE)
    assert clone.stoch.shape == (BATCH_SIZE, SEQ_LEN, CATEGORY_SIZE * CLASS_SIZE)
    assert clone.feature.shape == (
        BATCH_SIZE,
        SEQ_LEN,
        DETERMINISTIC_SIZE + CATEGORY_SIZE * CLASS_SIZE,
    )
    assert clone.deter.requires_grad is True
    assert clone.stoch.requires_grad is True


def test__unsqueeze(state: State) -> None:
    """Test the unsqueeze method."""
    state_unsqueezed = state.unsqueeze(dim=1)
    assert state_unsqueezed.deter.shape == (BATCH_SIZE, 1, SEQ_LEN, DETERMINISTIC_SIZE)
    assert state_unsqueezed.stoch.shape == (BATCH_SIZE, 1, SEQ_LEN, CATEGORY_SIZE * CLASS_SIZE)


def test__squeeze(state: State) -> None:
    """Test the squeeze method."""
    state_unsqueezed = state.unsqueeze(dim=1)
    state_squeezed = state_unsqueezed.squeeze(dim=1)
    assert state_squeezed.deter.shape == (BATCH_SIZE, SEQ_LEN, DETERMINISTIC_SIZE)
    assert state_squeezed.stoch.shape == (BATCH_SIZE, SEQ_LEN, CATEGORY_SIZE * CLASS_SIZE)


def test__stack_state(state: State) -> None:
    """Test the stack_states function."""
    num_states = 2
    states = [state] * num_states
    stacked = stack_states(states, dim=1)
    assert stacked.deter.shape == (
        BATCH_SIZE,
        num_states,
        SEQ_LEN,
        DETERMINISTIC_SIZE,
    )
    assert stacked.stoch.shape == (
        BATCH_SIZE,
        num_states,
        SEQ_LEN,
        CATEGORY_SIZE * CLASS_SIZE,
    )
    assert stacked.feature.shape == (
        BATCH_SIZE,
        num_states,
        SEQ_LEN,
        DETERMINISTIC_SIZE + CATEGORY_SIZE * CLASS_SIZE,
    )


def test__cat_state(state: State) -> None:
    """Test the cat_states function."""
    num_states = 2
    states = [state] * num_states
    concatenated = cat_states(states, dim=1)
    assert concatenated.deter.shape == (
        BATCH_SIZE,
        num_states * SEQ_LEN,
        DETERMINISTIC_SIZE,
    )
    assert concatenated.stoch.shape == (
        BATCH_SIZE,
        num_states * SEQ_LEN,
        CATEGORY_SIZE * CLASS_SIZE,
    )
    assert concatenated.feature.shape == (
        BATCH_SIZE,
        num_states * SEQ_LEN,
        DETERMINISTIC_SIZE + CATEGORY_SIZE * CLASS_SIZE,
    )
