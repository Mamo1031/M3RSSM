"""Tests for `m3rssm/state.py`."""

import pytest
import torch
from distribution_extension import MultiOneHot

from models.m3rssm.state import MTState, cat_mtstates, stack_mtstates
from tests.conftest import (
    BATCH_SIZE,
    CATEGORY_SIZE,
    CLASS_SIZE,
    DETERMINISTIC_SIZE,
    HIDDEN_SIZE,
    SEQ_LEN,
)


@pytest.fixture
def mtstate() -> MTState:
    """Create a MTState instance."""
    deter_h = torch.rand(BATCH_SIZE, SEQ_LEN, DETERMINISTIC_SIZE)
    deter_l = torch.rand(BATCH_SIZE, SEQ_LEN, DETERMINISTIC_SIZE)
    logit_h = torch.rand(BATCH_SIZE, SEQ_LEN, CATEGORY_SIZE, CLASS_SIZE)
    logit_l = torch.rand(BATCH_SIZE, SEQ_LEN, CATEGORY_SIZE, CLASS_SIZE)
    dist_h = MultiOneHot(logit_h)
    dist_l = MultiOneHot(logit_l)
    hidden_h = torch.rand(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
    hidden_l = torch.rand(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
    return MTState(
        deter_h=deter_h,
        deter_l=deter_l,
        distribution_h=dist_h,
        distribution_l=dist_l,
        hidden_h=hidden_h,
        hidden_l=hidden_l,
    )


def test_init(mtstate: MTState) -> None:
    """Test the __init__ method."""
    assert mtstate.deter_h.shape == (BATCH_SIZE, SEQ_LEN, DETERMINISTIC_SIZE)
    assert mtstate.deter_l.shape == (BATCH_SIZE, SEQ_LEN, DETERMINISTIC_SIZE)
    assert mtstate.stoch_h.shape == (BATCH_SIZE, SEQ_LEN, CATEGORY_SIZE * CLASS_SIZE)
    assert mtstate.stoch_l.shape == (BATCH_SIZE, SEQ_LEN, CATEGORY_SIZE * CLASS_SIZE)
    assert mtstate.feature.shape == (
        BATCH_SIZE,
        SEQ_LEN,
        2 * (DETERMINISTIC_SIZE + CATEGORY_SIZE * CLASS_SIZE),
    )


def test__iter__(mtstate: MTState) -> None:
    """Test the __iter__ method."""
    for s in mtstate:
        assert s.deter_h.shape == (SEQ_LEN, DETERMINISTIC_SIZE)
        assert s.deter_l.shape == (SEQ_LEN, DETERMINISTIC_SIZE)


def test__getitem__(mtstate: MTState) -> None:
    """Test the __getitem__ method."""
    s = mtstate[:, 0]
    assert s.deter_h.shape == (BATCH_SIZE, DETERMINISTIC_SIZE)
    assert s.deter_l.shape == (BATCH_SIZE, DETERMINISTIC_SIZE)


def test__to(mtstate: MTState) -> None:
    """Test the to method."""
    mtstate_moved = mtstate.to(torch.device("cpu"))
    assert mtstate_moved.deter_h.device == torch.device("cpu")
    assert mtstate_moved.deter_l.device == torch.device("cpu")


def test__detach(mtstate: MTState) -> None:
    """Test the detach method."""
    mtstate_detached = mtstate.detach()
    assert mtstate_detached.deter_h.requires_grad is False


def test__clone(mtstate: MTState) -> None:
    """Test the clone method."""
    clone = mtstate.clone()
    assert clone.deter_h.shape == mtstate.deter_h.shape
    assert clone.deter_l.shape == mtstate.deter_l.shape


def test__stack_mtstates(mtstate: MTState) -> None:
    """Test the stack_mtstates function."""
    num_states = 2
    states = [mtstate] * num_states
    stacked = stack_mtstates(states, dim=1)
    assert stacked.deter_h.shape == (
        BATCH_SIZE,
        num_states,
        SEQ_LEN,
        DETERMINISTIC_SIZE,
    )
    assert stacked.deter_l.shape == (
        BATCH_SIZE,
        num_states,
        SEQ_LEN,
        DETERMINISTIC_SIZE,
    )


def test__squeeze_unsqueeze_hidden_dim1() -> None:
    """Test squeeze/unsqueeze when hidden_h/hidden_l have dim 1 (edge case)."""
    deter_h = torch.rand(1, 3, DETERMINISTIC_SIZE)
    deter_l = torch.rand(1, 3, DETERMINISTIC_SIZE)
    logit_h = torch.rand(1, 3, CATEGORY_SIZE, CLASS_SIZE)
    logit_l = torch.rand(1, 3, CATEGORY_SIZE, CLASS_SIZE)
    dist_h = MultiOneHot(logit_h)
    dist_l = MultiOneHot(logit_l)
    hidden_h = torch.rand(1, 3, HIDDEN_SIZE)
    hidden_l = torch.rand(HIDDEN_SIZE)  # 1D: squeeze keeps as-is

    state = MTState(
        deter_h=deter_h,
        deter_l=deter_l,
        distribution_h=dist_h,
        distribution_l=dist_l,
        hidden_h=hidden_h,
        hidden_l=hidden_l,
    )
    squeezed = state.squeeze(0)
    assert squeezed.deter_h.shape == (3, DETERMINISTIC_SIZE)
    assert squeezed.hidden_l.shape == (HIDDEN_SIZE,)

    unsqueezed = squeezed.unsqueeze(0)
    assert unsqueezed.deter_h.shape == (1, 3, DETERMINISTIC_SIZE)


def test__cat_mtstates(mtstate: MTState) -> None:
    """Test the cat_mtstates function."""
    num_states = 2
    states = [mtstate] * num_states
    concatenated = cat_mtstates(states, dim=1)
    assert concatenated.deter_h.shape == (
        BATCH_SIZE,
        num_states * SEQ_LEN,
        DETERMINISTIC_SIZE,
    )
    assert concatenated.deter_l.shape == (
        BATCH_SIZE,
        num_states * SEQ_LEN,
        DETERMINISTIC_SIZE,
    )
