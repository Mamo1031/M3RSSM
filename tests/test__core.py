"""Tests of `core.py` BaseRSSM."""

import torch
from torch import Tensor, nn

from models.core import BaseRSSM
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
)


def _make_minimal_rssm() -> BaseRSSM:
    """Create minimal BaseRSSM implementation for testing base class methods."""
    representation = Representation(
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

    class MinimalRSSM(BaseRSSM):
        def encode_observation(self, observation: Tensor | tuple[Tensor, ...]) -> Tensor:
            obs = observation[0] if isinstance(observation, tuple) else observation
            if obs.dim() == 2:
                return obs[:, :OBS_EMBED_SIZE]
            return obs[..., :OBS_EMBED_SIZE]

        def decode_state(self, state: State) -> dict[str, Tensor]:
            return {"recon": state.deter[:, :, :OBS_EMBED_SIZE]}

        def compute_reconstruction_loss(
            self,
            reconstructions: dict[str, Tensor],
            targets: dict[str, Tensor],
        ) -> dict[str, Tensor]:
            return {"recon": (reconstructions["recon"] - targets["recon"]).pow(2).mean()}

        def get_observations_from_batch(self, batch: tuple[Tensor, ...]) -> Tensor:
            return batch[1]

        def get_initial_observation(self, observations: Tensor) -> Tensor:
            return observations[:, 0]

        def get_targets_from_batch(self, batch: tuple[Tensor, ...]) -> dict[str, Tensor]:
            return {"recon": batch[1]}

    return MinimalRSSM(
        representation=representation,
        transition=transition,
        init_proj=init_proj,
        kl_coeff=0.1,
        use_kl_balancing=False,
    )


def test__base_rssm_rollout_transition() -> None:
    """Test BaseRSSM.rollout_transition."""
    model = _make_minimal_rssm()
    actions = torch.rand(BATCH_SIZE, SEQ_LEN, ACTION_SIZE)
    obs = torch.rand(BATCH_SIZE, OBS_EMBED_SIZE)
    prev_state = model.initial_state(obs)

    prior = model.rollout_transition(actions=actions, prev_state=prev_state)
    assert prior.deter.shape == (BATCH_SIZE, SEQ_LEN, DETERMINISTIC_SIZE)


def test__base_rssm_shared_step() -> None:
    """Test BaseRSSM.shared_step."""
    model = _make_minimal_rssm()
    actions = torch.rand(BATCH_SIZE, SEQ_LEN, ACTION_SIZE)
    obs = torch.rand(BATCH_SIZE, SEQ_LEN, OBS_EMBED_SIZE)
    batch = (actions, obs)

    loss_dict = model.shared_step(batch)
    assert "loss" in loss_dict
    assert "recon" in loss_dict
    assert "kl" in loss_dict
