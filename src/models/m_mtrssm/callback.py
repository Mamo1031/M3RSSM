"""Callbacks for M_MTRSSM (3 modalities with early fusion, hierarchical states)."""

from torch import Tensor

from models.m3rssm.state import MTState, cat_mtstates
from models.m_mtrssm.core import M_MTRSSM
from models.mopoe_mrssm.callback import LogMultimodalMRSSMOutput


class LogM_MTRSSMOutput(LogMultimodalMRSSMOutput):  # noqa: N801
    """Log M_MTRSSM output (3 modalities, hierarchical MTState)."""

    def __init__(
        self,
        *,
        every_n_epochs: int,
        indices: list[int],
        query_length: int,
        fps: float,
    ) -> None:
        """Initialize LogM_MTRSSMOutput.

        Args:
            every_n_epochs: Log every N epochs.
            indices: Episode indices to log.
            query_length: Query length for rollout.
            fps: Frames per second for video logging.
        """
        super().__init__(
            every_n_epochs=every_n_epochs,
            indices=indices,
            query_length=query_length,
            fps=fps,
            model_types=(M_MTRSSM,),
        )

    def _compute_reconstructions(  # noqa: PLR0913, PLR0917
        self,
        model: M_MTRSSM,
        action_input: Tensor,
        observation_input: tuple[Tensor, Tensor, Tensor],
        vision_obs_input: Tensor,
        left_tactile_obs_input: Tensor,
        right_tactile_obs_input: Tensor,
    ) -> tuple[MTState, MTState]:
        """Compute posterior and prior reconstructions using MTState operations.

        Args:
            model: M_MTRSSM model instance.
            action_input: Action input tensor.
            observation_input: Tuple of (vision, left_tactile, right_tactile) inputs.
            vision_obs_input: Vision observation input.
            left_tactile_obs_input: Left tactile observation input.
            right_tactile_obs_input: Right tactile observation input.

        Returns
        -------
        tuple[MTState, MTState]: (posterior, prior) hierarchical states.
        """
        posterior, _ = model.rollout_representation(
            actions=action_input,
            observations=observation_input,
            prev_state=model.initial_state((
                vision_obs_input[:, 0],
                left_tactile_obs_input[:, 0],
                right_tactile_obs_input[:, 0],
            )),
        )

        prior = model.rollout_transition(
            actions=action_input[:, self.query_length :],
            prev_state=posterior[:, self.query_length - 1],
        )
        prior = cat_mtstates([posterior[:, : self.query_length], prior], dim=1)
        return posterior, prior
