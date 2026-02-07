"""Common constants and fixtures in the test."""

import pytest
import torch
from distribution_extension import MultiOneHot
from torch import Tensor

from models.m3rssm.state import MTState
from models.state import State

BATCH_SIZE = 4
SEQ_LEN = 8
DETERMINISTIC_SIZE = 64
CATEGORY_SIZE = 4
CLASS_SIZE = 4
ACTION_SIZE = 15
HIDDEN_SIZE = 32
OBS_EMBED_SIZE = 7

# Tactile observation shapes (simplified for testing)
TACTILE_SHAPE = (8, 8)
VISION_SHAPE = (3, 64, 64)


@pytest.fixture
def action_bd() -> Tensor:
    """Create a batch of actions."""
    return torch.rand(BATCH_SIZE, ACTION_SIZE)


@pytest.fixture
def observation_bd() -> Tensor:
    """Create a batch of observations."""
    return torch.rand(BATCH_SIZE, 3, 64, 64)


@pytest.fixture
def obs_embed_bd() -> Tensor:
    """Create a batch of observation embeddings."""
    return torch.rand(BATCH_SIZE, OBS_EMBED_SIZE)


@pytest.fixture
def state_discrete_bd() -> State:
    """Create a batch of states (discrete)."""
    deter = torch.rand(BATCH_SIZE, DETERMINISTIC_SIZE)
    logit = torch.rand(BATCH_SIZE, CATEGORY_SIZE, CLASS_SIZE)
    distribution = MultiOneHot(logit)
    return State(deter=deter, distribution=distribution)


@pytest.fixture
def action_bld() -> Tensor:
    """Create a batch of action sequences."""
    return torch.rand(BATCH_SIZE, SEQ_LEN, ACTION_SIZE)


@pytest.fixture
def observation_bld() -> Tensor:
    """Create a batch of vision observation sequences."""
    return torch.rand(BATCH_SIZE, SEQ_LEN, 3, 64, 64)


@pytest.fixture
def vision_obs_bd() -> Tensor:
    """Create a batch of vision observations (single timestep)."""
    return torch.rand(BATCH_SIZE, 3, 64, 64)


@pytest.fixture
def vision_obs_bld() -> Tensor:
    """Create a batch of vision observation sequences."""
    return torch.rand(BATCH_SIZE, SEQ_LEN, 3, 64, 64)


@pytest.fixture
def tactile_obs_bd() -> Tensor:
    """Create a batch of tactile observations (single timestep)."""
    return torch.rand(BATCH_SIZE, *TACTILE_SHAPE)


@pytest.fixture
def tactile_obs_bld() -> Tensor:
    """Create a batch of tactile observation sequences."""
    return torch.rand(BATCH_SIZE, SEQ_LEN, *TACTILE_SHAPE)


@pytest.fixture
def mtstate_bd() -> MTState:
    """Create a batch of MTStates (hierarchical)."""
    deter_h = torch.rand(BATCH_SIZE, DETERMINISTIC_SIZE)
    deter_l = torch.rand(BATCH_SIZE, DETERMINISTIC_SIZE)
    logit_h = torch.rand(BATCH_SIZE, CATEGORY_SIZE, CLASS_SIZE)
    logit_l = torch.rand(BATCH_SIZE, CATEGORY_SIZE, CLASS_SIZE)
    dist_h = MultiOneHot(logit_h)
    dist_l = MultiOneHot(logit_l)
    hidden_h = torch.rand(BATCH_SIZE, HIDDEN_SIZE)
    hidden_l = torch.rand(BATCH_SIZE, HIDDEN_SIZE)
    return MTState(
        deter_h=deter_h,
        deter_l=deter_l,
        distribution_h=dist_h,
        distribution_l=dist_l,
        hidden_h=hidden_h,
        hidden_l=hidden_l,
    )
