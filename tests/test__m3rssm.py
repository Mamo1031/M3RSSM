"""Tests of `m3rssm/core.py`."""

import pytest
import torch
from distribution_extension import MultiOneHotFactory
from torch import Tensor, nn, rand

from models.m3rssm.core import M3RSSM
from models.m3rssm.state import MTState
from models.networks import Representation
from tests.conftest import (
    ACTION_SIZE,
    BATCH_SIZE,
    CATEGORY_SIZE,
    CLASS_SIZE,
    HIDDEN_SIZE,
    OBS_EMBED_SIZE,
    SEQ_LEN,
    TACTILE_SHAPE,
)

# M3RSSM specific dimensions
HD_DIM = 8
HS_DIM = CLASS_SIZE * CATEGORY_SIZE  # 16
LD_DIM = 16
LS_DIM = CLASS_SIZE * CATEGORY_SIZE  # 16
L_TAU = 4.0
H_TAU = 8.0


class DummyVisionEncoder(nn.Module):
    """A dummy vision encoder for testing."""

    def forward(self, vision_obs: Tensor) -> Tensor:
        """Encode vision observation."""
        if vision_obs.ndim == 5:
            return rand(BATCH_SIZE, SEQ_LEN, OBS_EMBED_SIZE)
        return rand(BATCH_SIZE, OBS_EMBED_SIZE)


class DummyTactileEncoder(nn.Module):
    """A dummy tactile encoder for testing."""

    def forward(self, tactile_obs: Tensor) -> Tensor:
        """Encode tactile observation."""
        if tactile_obs.ndim == 4:
            return rand(BATCH_SIZE, SEQ_LEN, OBS_EMBED_SIZE)
        return rand(BATCH_SIZE, OBS_EMBED_SIZE)


class DummyVisionDecoder(nn.Module):
    """A dummy vision decoder for testing."""

    def forward(self, feature: Tensor) -> Tensor:
        """Decode feature."""
        if feature.ndim == 2:
            return rand(BATCH_SIZE, SEQ_LEN, 3, 64, 64)
        return rand(BATCH_SIZE, SEQ_LEN, 3, 64, 64)


class DummyTactileDecoder(nn.Module):
    """A dummy tactile decoder for testing."""

    def forward(self, feature: Tensor) -> Tensor:
        """Decode feature."""
        if feature.ndim == 2:
            return rand(BATCH_SIZE, SEQ_LEN, *TACTILE_SHAPE)
        return rand(BATCH_SIZE, SEQ_LEN, *TACTILE_SHAPE)


@pytest.fixture
def discrete_m3rssm() -> M3RSSM:
    """Create a discrete M3RSSM instance."""
    vision_representation = Representation(
        deterministic_size=LD_DIM,
        hidden_size=HIDDEN_SIZE,
        obs_embed_size=OBS_EMBED_SIZE,
        distribution_config=(CLASS_SIZE, CATEGORY_SIZE),
        activation_name="ReLU",
    )
    left_tactile_representation = Representation(
        deterministic_size=LD_DIM,
        hidden_size=HIDDEN_SIZE,
        obs_embed_size=OBS_EMBED_SIZE,
        distribution_config=(CLASS_SIZE, CATEGORY_SIZE),
        activation_name="ReLU",
    )
    right_tactile_representation = Representation(
        deterministic_size=LD_DIM,
        hidden_size=HIDDEN_SIZE,
        obs_embed_size=OBS_EMBED_SIZE,
        distribution_config=(CLASS_SIZE, CATEGORY_SIZE),
        activation_name="ReLU",
    )
    init_proj = nn.Linear(OBS_EMBED_SIZE, HD_DIM + LD_DIM)
    l_prior = nn.Linear(LD_DIM, LS_DIM)
    l_posterior = nn.Linear(LD_DIM + OBS_EMBED_SIZE, HS_DIM)
    h_prior = nn.Linear(HD_DIM, HS_DIM)
    h_posterior = nn.Linear(HD_DIM + HS_DIM, HS_DIM)
    l_dist = MultiOneHotFactory(class_size=CLASS_SIZE, category_size=CATEGORY_SIZE)
    h_dist = MultiOneHotFactory(class_size=CLASS_SIZE, category_size=CATEGORY_SIZE)

    return M3RSSM(
        vision_representation=vision_representation,
        left_tactile_representation=left_tactile_representation,
        right_tactile_representation=right_tactile_representation,
        vision_encoder=DummyVisionEncoder(),
        tactile_encoder=DummyTactileEncoder(),
        vision_decoder=DummyVisionDecoder(),
        left_tactile_decoder=DummyTactileDecoder(),
        right_tactile_decoder=DummyTactileDecoder(),
        init_proj=init_proj,
        kl_coeff=1.0,
        use_kl_balancing=True,
        action_size=ACTION_SIZE,
        hd_dim=HD_DIM,
        hs_dim=HS_DIM,
        ld_dim=LD_DIM,
        ls_dim=LS_DIM,
        l_tau=L_TAU,
        h_tau=H_TAU,
        l_prior=l_prior,
        l_posterior=l_posterior,
        h_prior=h_prior,
        h_posterior=h_posterior,
        l_dist=l_dist,
        h_dist=h_dist,
    )


@pytest.fixture
def m3rssm_mtstate(discrete_m3rssm: M3RSSM) -> MTState:
    """Create MTState with dimensions matching M3RSSM."""
    vision_obs = torch.rand(BATCH_SIZE, 3, 64, 64)
    tactile_obs = torch.rand(BATCH_SIZE, *TACTILE_SHAPE)
    observation = (vision_obs, tactile_obs, tactile_obs)
    return discrete_m3rssm.initial_state(observation)


def test__initial_state(
    discrete_m3rssm: M3RSSM,
    vision_obs_bd: Tensor,
    tactile_obs_bd: Tensor,
) -> None:
    """Test `initial_state` method."""
    observation = (vision_obs_bd, tactile_obs_bd, tactile_obs_bd)
    state = discrete_m3rssm.initial_state(observation)
    assert isinstance(state, MTState)
    assert state.deter_h.shape == (BATCH_SIZE, HD_DIM)
    assert state.deter_l.shape == (BATCH_SIZE, LD_DIM)


def test__rollout_representation(
    action_bld: Tensor,
    vision_obs_bld: Tensor,
    tactile_obs_bld: Tensor,
    m3rssm_mtstate: MTState,
    discrete_m3rssm: M3RSSM,
) -> None:
    """Test `rollout_representation` method."""
    observations = (vision_obs_bld, tactile_obs_bld, tactile_obs_bld)
    posterior, prior = discrete_m3rssm.rollout_representation(
        actions=action_bld,
        observations=observations,
        prev_state=m3rssm_mtstate,
    )
    assert isinstance(posterior, MTState)
    assert isinstance(prior, MTState)
    assert prior.deter_h.shape == (BATCH_SIZE, SEQ_LEN, HD_DIM)
    assert prior.deter_l.shape == (BATCH_SIZE, SEQ_LEN, LD_DIM)


def test__rollout_representation_invalid_observations(
    action_bld: Tensor,
    m3rssm_mtstate: MTState,
    discrete_m3rssm: M3RSSM,
) -> None:
    """Test `rollout_representation` with invalid observations (not a tuple)."""
    invalid_observations = rand(BATCH_SIZE, SEQ_LEN, 3, 64, 64)
    with pytest.raises(TypeError, match="M3RSSM requires tuple"):
        discrete_m3rssm.rollout_representation(
            actions=action_bld,
            observations=invalid_observations,
            prev_state=m3rssm_mtstate,
        )


def test__get_observations_from_batch(
    action_bld: Tensor,
    vision_obs_bld: Tensor,
    tactile_obs_bld: Tensor,
    discrete_m3rssm: M3RSSM,
) -> None:
    """Test `get_observations_from_batch` method."""
    batch = (
        action_bld,
        vision_obs_bld,
        tactile_obs_bld,
        tactile_obs_bld,
        action_bld,
        vision_obs_bld,
        tactile_obs_bld,
        tactile_obs_bld,
    )
    observations = discrete_m3rssm.get_observations_from_batch(batch)
    assert isinstance(observations, tuple)
    assert len(observations) == 3
    vision_obs, left_tactile_obs, right_tactile_obs = observations
    assert vision_obs.shape == vision_obs_bld.shape
    assert left_tactile_obs.shape == tactile_obs_bld.shape
    assert right_tactile_obs.shape == tactile_obs_bld.shape


def test__get_targets_from_batch(
    action_bld: Tensor,
    vision_obs_bld: Tensor,
    tactile_obs_bld: Tensor,
    discrete_m3rssm: M3RSSM,
) -> None:
    """Test `get_targets_from_batch` method."""
    batch = (
        action_bld,
        vision_obs_bld,
        tactile_obs_bld,
        tactile_obs_bld,
        action_bld,
        vision_obs_bld,
        tactile_obs_bld,
        tactile_obs_bld,
    )
    targets = discrete_m3rssm.get_targets_from_batch(batch)
    assert "recon/vision" in targets
    assert "recon/left_tactile" in targets
    assert "recon/right_tactile" in targets


def test__training_step(
    action_bld: Tensor,
    vision_obs_bld: Tensor,
    tactile_obs_bld: Tensor,
    discrete_m3rssm: M3RSSM,
) -> None:
    """Test `training_step` method."""
    batch = (
        action_bld,
        vision_obs_bld,
        tactile_obs_bld,
        tactile_obs_bld,
        action_bld,
        vision_obs_bld,
        tactile_obs_bld,
        tactile_obs_bld,
    )
    loss = discrete_m3rssm.training_step(batch, 0)
    assert "loss" in loss
    assert "train/kl" in loss
    assert "train/recon" in loss


def test__validation_step(
    action_bld: Tensor,
    vision_obs_bld: Tensor,
    tactile_obs_bld: Tensor,
    discrete_m3rssm: M3RSSM,
) -> None:
    """Test `validation_step` method."""
    batch = (
        action_bld,
        vision_obs_bld,
        tactile_obs_bld,
        tactile_obs_bld,
        action_bld,
        vision_obs_bld,
        tactile_obs_bld,
        tactile_obs_bld,
    )
    loss = discrete_m3rssm.validation_step(batch, 0)
    assert "val/loss" in loss
    assert "val/kl" in loss
    assert "val/recon" in loss


def test__rollout_transition(
    action_bld: Tensor,
    m3rssm_mtstate: MTState,
    discrete_m3rssm: M3RSSM,
) -> None:
    """Test `rollout_transition` method."""
    result = discrete_m3rssm.rollout_transition(
        actions=action_bld,
        prev_state=m3rssm_mtstate,
    )
    assert isinstance(result, MTState)
    assert result.deter_h.shape == (BATCH_SIZE, SEQ_LEN, 8)
    assert result.deter_l.shape == (BATCH_SIZE, SEQ_LEN, 16)


def test__decode_state(
    m3rssm_mtstate: MTState,
    discrete_m3rssm: M3RSSM,
) -> None:
    """Test `decode_state` method."""
    state_seq = discrete_m3rssm.rollout_transition(
        actions=torch.rand(BATCH_SIZE, 1, ACTION_SIZE),
        prev_state=m3rssm_mtstate,
    )
    recon = discrete_m3rssm.decode_state(state_seq[:, 0])
    assert "recon/vision" in recon
    assert "recon/left_tactile" in recon
    assert "recon/right_tactile" in recon
