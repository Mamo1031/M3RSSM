"""Tests of `mopoe_mrssm/core.py`."""

import pytest
from torch import Tensor, nn, rand
from torch.nn import functional

from models.mopoe_mrssm.core import MoPoE_MRSSM
from models.networks import Representation, Transition
from models.state import State
from tests.conftest import (
    ACTION_SIZE,
    BATCH_SIZE,
    CATEGORY_SIZE,
    CLASS_SIZE,
    DETERMINISTIC_SIZE,
    HIDDEN_SIZE,
    OBS_EMBED_SIZE,
    SEQ_LEN,
    TACTILE_SHAPE,
)


class DummyTactileEncoder(nn.Module):
    """A dummy tactile encoder for testing."""

    def forward(self, tactile_obs: Tensor) -> Tensor:
        """Encode tactile observation."""
        if tactile_obs.ndim == 4:
            return rand(BATCH_SIZE, SEQ_LEN, OBS_EMBED_SIZE)
        return rand(BATCH_SIZE, OBS_EMBED_SIZE)


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


class DummyTactileDecoder(nn.Module):
    """A dummy tactile decoder for testing."""

    def forward(self, feature: Tensor) -> Tensor:
        """Decode feature."""
        if feature.ndim == 2:
            return rand(BATCH_SIZE, SEQ_LEN, *TACTILE_SHAPE)
        return rand(BATCH_SIZE, SEQ_LEN, *TACTILE_SHAPE)


@pytest.fixture
def discrete_mopoe_mrssm() -> MoPoE_MRSSM:
    """Create a discrete MoPoE-MRSSM instance."""
    vision_representation = Representation(
        deterministic_size=DETERMINISTIC_SIZE,
        hidden_size=HIDDEN_SIZE,
        obs_embed_size=OBS_EMBED_SIZE,
        distribution_config=(CLASS_SIZE, CATEGORY_SIZE),
        activation_name="ReLU",
    )
    left_tactile_representation = Representation(
        deterministic_size=DETERMINISTIC_SIZE,
        hidden_size=HIDDEN_SIZE,
        obs_embed_size=OBS_EMBED_SIZE,
        distribution_config=(CLASS_SIZE, CATEGORY_SIZE),
        activation_name="ReLU",
    )
    right_tactile_representation = Representation(
        deterministic_size=DETERMINISTIC_SIZE,
        hidden_size=HIDDEN_SIZE,
        obs_embed_size=OBS_EMBED_SIZE,
        distribution_config=(CLASS_SIZE, CATEGORY_SIZE),
        activation_name="ReLU",
    )
    transition = Transition(
        action_size=ACTION_SIZE,
        deterministic_size=DETERMINISTIC_SIZE,
        hidden_size=HIDDEN_SIZE,
        distribution_config=(CLASS_SIZE, CATEGORY_SIZE),
        activation_name="ReLU",
    )
    init_proj = nn.Linear(OBS_EMBED_SIZE, DETERMINISTIC_SIZE)
    return MoPoE_MRSSM(
        vision_representation=vision_representation,
        left_tactile_representation=left_tactile_representation,
        right_tactile_representation=right_tactile_representation,
        transition=transition,
        vision_encoder=DummyVisionEncoder(),
        tactile_encoder=DummyTactileEncoder(),
        vision_decoder=DummyVisionDecoder(),
        left_tactile_decoder=DummyTactileDecoder(),
        right_tactile_decoder=DummyTactileDecoder(),
        init_proj=init_proj,
        kl_coeff=1.0,
        use_kl_balancing=True,
    )


def test__initial_state(
    discrete_mopoe_mrssm: MoPoE_MRSSM,
    vision_obs_bd: Tensor,
    tactile_obs_bd: Tensor,
) -> None:
    """Test `initial_state` method."""
    observation = (vision_obs_bd, tactile_obs_bd, tactile_obs_bd)
    state = discrete_mopoe_mrssm.initial_state(observation)  # type: ignore[arg-type]
    assert isinstance(state, State)
    assert state.deter.shape == (BATCH_SIZE, DETERMINISTIC_SIZE)
    assert state.stoch.shape == (BATCH_SIZE, CATEGORY_SIZE * CLASS_SIZE)


def test__rollout_representation(
    action_bld: Tensor,
    vision_obs_bld: Tensor,
    tactile_obs_bld: Tensor,
    state_discrete_bd: State,
    discrete_mopoe_mrssm: MoPoE_MRSSM,
) -> None:
    """Test `rollout_representation` method."""
    observations = (vision_obs_bld, tactile_obs_bld, tactile_obs_bld)
    posterior, prior = discrete_mopoe_mrssm.rollout_representation(
        actions=action_bld,
        observations=observations,
        prev_state=state_discrete_bd,
    )
    feature_size = CATEGORY_SIZE * CLASS_SIZE
    assert prior.deter.shape == (BATCH_SIZE, SEQ_LEN, DETERMINISTIC_SIZE)
    assert prior.stoch.shape == (BATCH_SIZE, SEQ_LEN, feature_size)
    assert posterior.deter.shape == (BATCH_SIZE, SEQ_LEN, DETERMINISTIC_SIZE)
    assert posterior.stoch.shape == (BATCH_SIZE, SEQ_LEN, feature_size)


def test__rollout_representation_invalid_observations(
    action_bld: Tensor,
    state_discrete_bd: State,
    discrete_mopoe_mrssm: MoPoE_MRSSM,
) -> None:
    """Test `rollout_representation` with invalid observations (not a tuple)."""
    invalid_observations = rand(BATCH_SIZE, SEQ_LEN, 3, 64, 64)
    with pytest.raises(TypeError, match="MoPoE-MRSSM requires tuple"):
        discrete_mopoe_mrssm.rollout_representation(
            actions=action_bld,
            observations=invalid_observations,
            prev_state=state_discrete_bd,
        )


def test__get_observations_from_batch(
    action_bld: Tensor,
    vision_obs_bld: Tensor,
    tactile_obs_bld: Tensor,
    discrete_mopoe_mrssm: MoPoE_MRSSM,
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
    observations = discrete_mopoe_mrssm.get_observations_from_batch(batch)
    assert isinstance(observations, tuple)
    assert len(observations) == 3
    vision_obs, left_tactile_obs, right_tactile_obs = observations
    assert vision_obs.shape == vision_obs_bld.shape
    assert left_tactile_obs.shape == tactile_obs_bld.shape
    assert right_tactile_obs.shape == tactile_obs_bld.shape


def test__get_initial_observation(
    vision_obs_bld: Tensor,
    tactile_obs_bld: Tensor,
    discrete_mopoe_mrssm: MoPoE_MRSSM,
) -> None:
    """Test `get_initial_observation` method."""
    observations = (vision_obs_bld, tactile_obs_bld, tactile_obs_bld)
    initial_obs = discrete_mopoe_mrssm.get_initial_observation(observations)
    assert isinstance(initial_obs, tuple)
    assert len(initial_obs) == 3
    vision_obs, left_tactile_obs, right_tactile_obs = initial_obs
    assert vision_obs.shape == (BATCH_SIZE, 3, 64, 64)
    assert left_tactile_obs.shape == (BATCH_SIZE, *TACTILE_SHAPE)
    assert right_tactile_obs.shape == (BATCH_SIZE, *TACTILE_SHAPE)


def test__get_targets_from_batch(
    action_bld: Tensor,
    vision_obs_bld: Tensor,
    tactile_obs_bld: Tensor,
    discrete_mopoe_mrssm: MoPoE_MRSSM,
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
    targets = discrete_mopoe_mrssm.get_targets_from_batch(batch)
    assert "recon/vision" in targets
    assert "recon/left_tactile" in targets
    assert "recon/right_tactile" in targets
    assert targets["recon/vision"].shape == vision_obs_bld.shape


def test__training_step(
    action_bld: Tensor,
    vision_obs_bld: Tensor,
    tactile_obs_bld: Tensor,
    discrete_mopoe_mrssm: MoPoE_MRSSM,
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
    loss = discrete_mopoe_mrssm.training_step(batch, 0)
    assert "loss" in loss
    assert "train/kl" in loss
    assert "train/recon" in loss
    assert "train/recon/vision" in loss


def test__validation_step(
    action_bld: Tensor,
    vision_obs_bld: Tensor,
    tactile_obs_bld: Tensor,
    discrete_mopoe_mrssm: MoPoE_MRSSM,
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
    loss = discrete_mopoe_mrssm.validation_step(batch, 0)
    assert "val/loss" in loss
    assert "val/kl" in loss
    assert "val/recon" in loss


def test__poe_fusion_categorical_2mod(
    discrete_mopoe_mrssm: MoPoE_MRSSM,
    state_discrete_bd: State,
) -> None:
    """Test `_poe_fusion_categorical_2mod` method."""
    logits1 = rand(BATCH_SIZE, CLASS_SIZE * CATEGORY_SIZE)
    logits2 = rand(BATCH_SIZE, CLASS_SIZE * CATEGORY_SIZE)
    fused = discrete_mopoe_mrssm._poe_fusion_categorical_2mod(
        logits1=logits1,
        logits2=logits2,
        prior_state=state_discrete_bd,
    )
    assert fused.deter.shape == (BATCH_SIZE, DETERMINISTIC_SIZE)
    assert fused.stoch.shape == (BATCH_SIZE, CATEGORY_SIZE * CLASS_SIZE)


def test__poe_fusion_categorical_3mod(
    discrete_mopoe_mrssm: MoPoE_MRSSM,
    state_discrete_bd: State,
) -> None:
    """Test `_poe_fusion_categorical_3mod` method."""
    logits1 = rand(BATCH_SIZE, CLASS_SIZE * CATEGORY_SIZE)
    logits2 = rand(BATCH_SIZE, CLASS_SIZE * CATEGORY_SIZE)
    logits3 = rand(BATCH_SIZE, CLASS_SIZE * CATEGORY_SIZE)
    fused = discrete_mopoe_mrssm._poe_fusion_categorical_3mod(
        logits1=logits1,
        logits2=logits2,
        logits3=logits3,
        prior_state=state_discrete_bd,
    )
    assert fused.deter.shape == (BATCH_SIZE, DETERMINISTIC_SIZE)
    assert fused.stoch.shape == (BATCH_SIZE, CATEGORY_SIZE * CLASS_SIZE)


def test__moe_fusion_categorical(
    discrete_mopoe_mrssm: MoPoE_MRSSM,
    state_discrete_bd: State,
) -> None:
    """Test `_moe_fusion_categorical` method."""
    logits = rand(BATCH_SIZE, CLASS_SIZE * CATEGORY_SIZE)
    prior_log_probs = functional.log_softmax(logits, dim=-1)
    vision_log_probs = functional.log_softmax(rand(BATCH_SIZE, CLASS_SIZE * CATEGORY_SIZE), dim=-1)
    left_log_probs = functional.log_softmax(rand(BATCH_SIZE, CLASS_SIZE * CATEGORY_SIZE), dim=-1)
    right_log_probs = functional.log_softmax(rand(BATCH_SIZE, CLASS_SIZE * CATEGORY_SIZE), dim=-1)
    vl_poe = vision_log_probs + left_log_probs
    vr_poe = vision_log_probs + right_log_probs
    lr_poe = left_log_probs + right_log_probs
    vlr_poe = vision_log_probs + left_log_probs + right_log_probs

    mixed = discrete_mopoe_mrssm._moe_fusion_categorical(
        prior_log_probs=prior_log_probs,
        vision_log_probs=vision_log_probs,
        left_tactile_log_probs=left_log_probs,
        right_tactile_log_probs=right_log_probs,
        vl_poe_log_probs=vl_poe,
        vr_poe_log_probs=vr_poe,
        lr_poe_log_probs=lr_poe,
        vlr_poe_log_probs=vlr_poe,
        prior_state=state_discrete_bd,
    )
    assert mixed.deter.shape == (BATCH_SIZE, DETERMINISTIC_SIZE)
    assert mixed.stoch.shape == (BATCH_SIZE, CATEGORY_SIZE * CLASS_SIZE)


def test__decode_state(
    state_discrete_bd: State,
    discrete_mopoe_mrssm: MoPoE_MRSSM,
) -> None:
    """Test `decode_state` method."""
    recon = discrete_mopoe_mrssm.decode_state(state_discrete_bd)
    assert "recon/vision" in recon
    assert "recon/left_tactile" in recon
    assert "recon/right_tactile" in recon


def test__encode_observation_tensor(discrete_mopoe_mrssm: MoPoE_MRSSM) -> None:
    """Test `encode_observation` with Tensor (backward compatibility)."""
    obs_embed = rand(BATCH_SIZE, OBS_EMBED_SIZE)
    result = discrete_mopoe_mrssm.encode_observation(obs_embed)
    assert result.shape == obs_embed.shape
