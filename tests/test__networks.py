"""Tests of `networks.py`."""

import pytest
from torch import Tensor

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
)


def test__representation(
    obs_embed_bd: Tensor,
    state_discrete_bd: State,
) -> None:
    """Test the Representation class and `forward()` method."""
    representation = Representation(
        deterministic_size=DETERMINISTIC_SIZE,
        hidden_size=HIDDEN_SIZE,
        obs_embed_size=OBS_EMBED_SIZE,
        distribution_config=(CLASS_SIZE, CATEGORY_SIZE),
        activation_name="ReLU",
    )
    posterior = representation.forward(
        obs_embed=obs_embed_bd,
        prior_state=state_discrete_bd,
    )
    assert posterior.deter.shape == (BATCH_SIZE, DETERMINISTIC_SIZE)
    assert posterior.stoch.shape == (BATCH_SIZE, CATEGORY_SIZE * CLASS_SIZE)


def test__transition(action_bd: Tensor, state_discrete_bd: State) -> None:
    """Test the Transition class and `forward()` method."""
    transition = Transition(
        action_size=ACTION_SIZE,
        deterministic_size=DETERMINISTIC_SIZE,
        hidden_size=HIDDEN_SIZE,
        distribution_config=(CLASS_SIZE, CATEGORY_SIZE),
        activation_name="ReLU",
    )
    prior = transition.forward(
        action=action_bd,
        prev_state=state_discrete_bd,
    )
    assert prior.deter.shape == (BATCH_SIZE, DETERMINISTIC_SIZE)
    assert prior.stoch.shape == (BATCH_SIZE, CATEGORY_SIZE * CLASS_SIZE)


def test__representation_with_list_config(
    obs_embed_bd: Tensor,
    state_discrete_bd: State,
) -> None:
    """Test Representation class with list distribution_config (YAML compatibility)."""
    representation = Representation(
        deterministic_size=DETERMINISTIC_SIZE,
        hidden_size=HIDDEN_SIZE,
        obs_embed_size=OBS_EMBED_SIZE,
        distribution_config=[CLASS_SIZE, CATEGORY_SIZE],
        activation_name="ReLU",
    )
    posterior = representation.forward(
        obs_embed=obs_embed_bd,
        prior_state=state_discrete_bd,
    )
    assert posterior.deter.shape == (BATCH_SIZE, DETERMINISTIC_SIZE)
    assert posterior.stoch.shape == (BATCH_SIZE, CATEGORY_SIZE * CLASS_SIZE)


def test__representation_invalid_list_config() -> None:
    """Test Representation class with invalid list distribution_config."""
    with pytest.raises(ValueError, match="must have 2 elements"):
        Representation(
            deterministic_size=DETERMINISTIC_SIZE,
            hidden_size=HIDDEN_SIZE,
            obs_embed_size=OBS_EMBED_SIZE,
            distribution_config=[CLASS_SIZE],
            activation_name="ReLU",
        )


def test__transition_invalid_list_config() -> None:
    """Test Transition class with invalid list distribution_config."""
    with pytest.raises(ValueError, match="must have 2 elements"):
        Transition(
            action_size=ACTION_SIZE,
            deterministic_size=DETERMINISTIC_SIZE,
            hidden_size=HIDDEN_SIZE,
            distribution_config=[CLASS_SIZE, CATEGORY_SIZE, 1],
            activation_name="ReLU",
        )


def test__transition_with_list_config(action_bd: Tensor, state_discrete_bd: State) -> None:
    """Test Transition class with list distribution_config (YAML compatibility)."""
    transition = Transition(
        action_size=ACTION_SIZE,
        deterministic_size=DETERMINISTIC_SIZE,
        hidden_size=HIDDEN_SIZE,
        distribution_config=[CLASS_SIZE, CATEGORY_SIZE],
        activation_name="ReLU",
    )
    prior = transition.forward(
        action=action_bd,
        prev_state=state_discrete_bd,
    )
    assert prior.deter.shape == (BATCH_SIZE, DETERMINISTIC_SIZE)
    assert prior.stoch.shape == (BATCH_SIZE, CATEGORY_SIZE * CLASS_SIZE)


def test__representation_with_different_activation(
    obs_embed_bd: Tensor,
    state_discrete_bd: State,
) -> None:
    """Test Representation class with different activation functions."""
    for activation_name in ["ReLU", "Tanh", "Sigmoid", "ELU"]:
        representation = Representation(
            deterministic_size=DETERMINISTIC_SIZE,
            hidden_size=HIDDEN_SIZE,
            obs_embed_size=OBS_EMBED_SIZE,
            distribution_config=(CLASS_SIZE, CATEGORY_SIZE),
            activation_name=activation_name,
        )
        posterior = representation.forward(
            obs_embed=obs_embed_bd,
            prior_state=state_discrete_bd,
        )
        assert posterior.deter.shape == (BATCH_SIZE, DETERMINISTIC_SIZE)
        assert posterior.stoch.shape == (BATCH_SIZE, CATEGORY_SIZE * CLASS_SIZE)


def test__transition_with_different_activation(action_bd: Tensor, state_discrete_bd: State) -> None:
    """Test Transition class with different activation functions."""
    for activation_name in ["ReLU", "Tanh", "Sigmoid", "ELU"]:
        transition = Transition(
            action_size=ACTION_SIZE,
            deterministic_size=DETERMINISTIC_SIZE,
            hidden_size=HIDDEN_SIZE,
            distribution_config=(CLASS_SIZE, CATEGORY_SIZE),
            activation_name=activation_name,
        )
        prior = transition.forward(
            action=action_bd,
            prev_state=state_discrete_bd,
        )
        assert prior.deter.shape == (BATCH_SIZE, DETERMINISTIC_SIZE)
        assert prior.stoch.shape == (BATCH_SIZE, CATEGORY_SIZE * CLASS_SIZE)
