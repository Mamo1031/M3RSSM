"""Tests of `mtrssm/core.py`."""

import pytest
import torch
from distribution_extension import MultiOneHotFactory
from torch import Tensor, nn, rand

from models.m3rssm.state import MTState
from models.mtrssm.core import MTRSSM
from models.networks import Representation
from tests.conftest import (
    ACTION_SIZE,
    BATCH_SIZE,
    CATEGORY_SIZE,
    CLASS_SIZE,
    HIDDEN_SIZE,
    OBS_EMBED_SIZE,
    SEQ_LEN,
)

# MTRSSM specific dimensions (hs_dim and ls_dim must match class_size * category_size)
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


class DummyVisionDecoder(nn.Module):
    """A dummy vision decoder for testing."""

    def forward(self, feature: Tensor) -> Tensor:
        """Decode feature."""
        if feature.ndim == 2:
            return rand(BATCH_SIZE, SEQ_LEN, 3, 64, 64)
        return rand(BATCH_SIZE, SEQ_LEN, 3, 64, 64)


@pytest.fixture
def discrete_mtrssm() -> MTRSSM:
    """Create a discrete MTRSSM instance."""
    vision_representation = Representation(
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

    return MTRSSM(
        vision_representation=vision_representation,
        vision_encoder=DummyVisionEncoder(),
        vision_decoder=DummyVisionDecoder(),
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


def test__initial_state(
    discrete_mtrssm: MTRSSM,
    vision_obs_bd: Tensor,
) -> None:
    """Test `initial_state` method."""
    state = discrete_mtrssm.initial_state(vision_obs_bd)
    assert isinstance(state, MTState)
    assert state.deter_h.shape == (BATCH_SIZE, HD_DIM)
    assert state.deter_l.shape == (BATCH_SIZE, LD_DIM)


@pytest.fixture
def mtrssm_mtstate(discrete_mtrssm: MTRSSM) -> MTState:
    """Create MTState with dimensions matching MTRSSM."""
    vision_obs = torch.rand(BATCH_SIZE, 3, 64, 64)
    return discrete_mtrssm.initial_state(vision_obs)


def test__rollout_representation(
    action_bld: Tensor,
    vision_obs_bld: Tensor,
    mtrssm_mtstate: MTState,
    discrete_mtrssm: MTRSSM,
) -> None:
    """Test `rollout_representation` method."""
    posterior, prior = discrete_mtrssm.rollout_representation(
        actions=action_bld,
        observations=vision_obs_bld,
        prev_state=mtrssm_mtstate,
    )
    assert isinstance(posterior, MTState)
    assert isinstance(prior, MTState)
    assert prior.deter_h.shape == (BATCH_SIZE, SEQ_LEN, HD_DIM)
    assert prior.deter_l.shape == (BATCH_SIZE, SEQ_LEN, LD_DIM)


def test__get_observations_from_batch(
    action_bld: Tensor,
    vision_obs_bld: Tensor,
    discrete_mtrssm: MTRSSM,
) -> None:
    """Test `get_observations_from_batch` method."""
    batch = (
        action_bld,
        vision_obs_bld,
        action_bld,
        vision_obs_bld,
    )
    observations = discrete_mtrssm.get_observations_from_batch(batch)
    assert observations.shape == vision_obs_bld.shape


def test__get_initial_observation(
    vision_obs_bld: Tensor,
    discrete_mtrssm: MTRSSM,
) -> None:
    """Test `get_initial_observation` method."""
    initial_obs = discrete_mtrssm.get_initial_observation(vision_obs_bld)
    assert initial_obs.shape == (BATCH_SIZE, 3, 64, 64)


def test__get_targets_from_batch(
    action_bld: Tensor,
    vision_obs_bld: Tensor,
    discrete_mtrssm: MTRSSM,
) -> None:
    """Test `get_targets_from_batch` method."""
    batch = (
        action_bld,
        vision_obs_bld,
        action_bld,
        vision_obs_bld,
    )
    targets = discrete_mtrssm.get_targets_from_batch(batch)
    assert "recon/vision" in targets
    assert targets["recon/vision"].shape == vision_obs_bld.shape


def test__training_step(
    action_bld: Tensor,
    vision_obs_bld: Tensor,
    discrete_mtrssm: MTRSSM,
) -> None:
    """Test `training_step` method."""
    batch = (
        action_bld,
        vision_obs_bld,
        action_bld,
        vision_obs_bld,
    )
    loss = discrete_mtrssm.training_step(batch, 0)
    assert "loss" in loss
    assert "train/kl" in loss
    assert "train/recon" in loss
    assert "train/recon/vision" in loss


def test__validation_step(
    action_bld: Tensor,
    vision_obs_bld: Tensor,
    discrete_mtrssm: MTRSSM,
) -> None:
    """Test `validation_step` method."""
    batch = (
        action_bld,
        vision_obs_bld,
        action_bld,
        vision_obs_bld,
    )
    loss = discrete_mtrssm.validation_step(batch, 0)
    assert "val/loss" in loss
    assert "val/kl" in loss
    assert "val/recon" in loss


def test__rollout_transition(
    action_bld: Tensor,
    mtrssm_mtstate: MTState,
    discrete_mtrssm: MTRSSM,
) -> None:
    """Test `rollout_transition` method."""
    result = discrete_mtrssm.rollout_transition(
        actions=action_bld,
        prev_state=mtrssm_mtstate,
    )
    assert isinstance(result, MTState)
    assert result.deter_h.shape == (BATCH_SIZE, SEQ_LEN, HD_DIM)
    assert result.deter_l.shape == (BATCH_SIZE, SEQ_LEN, LD_DIM)


def test__decode_state(
    mtrssm_mtstate: MTState,
    discrete_mtrssm: MTRSSM,
) -> None:
    """Test `decode_state` method."""
    state_seq = discrete_mtrssm.rollout_transition(
        actions=torch.rand(BATCH_SIZE, 1, ACTION_SIZE),
        prev_state=mtrssm_mtstate,
    )
    recon = discrete_mtrssm.decode_state(state_seq[:, 0])
    assert "recon/vision" in recon
    assert recon["recon/vision"].shape[0] == BATCH_SIZE
    assert recon["recon/vision"].shape[-3:] == (3, 64, 64)


def test__compute_reconstruction_loss(
    vision_obs_bld: Tensor,
    discrete_mtrssm: MTRSSM,
) -> None:
    """Test `compute_reconstruction_loss` static method."""
    reconstructions = {"recon/vision": vision_obs_bld, "recon": vision_obs_bld}
    targets = {"recon/vision": vision_obs_bld}
    loss_dict = discrete_mtrssm.compute_reconstruction_loss(reconstructions, targets)
    assert "recon" in loss_dict
    assert "recon/vision" in loss_dict
    assert loss_dict["recon"].shape == ()
