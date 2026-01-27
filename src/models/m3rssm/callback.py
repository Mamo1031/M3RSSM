"""Callbacks for M3RSSM."""

import torch
from lightning import LightningModule

from models.m3rssm.core import M3RSSM
from models.m3rssm.state import MTState, cat_mtstates
from models.mopoe_mrssm.callback import LogMultimodalMRSSMOutput


class LogM3RSSMOutput(LogMultimodalMRSSMOutput):
    """Log M3RSSM output with hierarchical states."""

    def __init__(
        self,
        *,
        every_n_epochs: int,
        indices: list[int],
        query_length: int,
        fps: float,
    ) -> None:
        """Initialize LogM3RSSMOutput."""
        super().__init__(
            every_n_epochs=every_n_epochs,
            indices=indices,
            query_length=query_length,
            fps=fps,
            model_types=(M3RSSM,),
        )

    def _compute_reconstructions(
        self,
        model: LightningModule,
        action_input: torch.Tensor,
        observation_input: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        vision_obs_input: torch.Tensor,
        left_tactile_obs_input: torch.Tensor,
        right_tactile_obs_input: torch.Tensor,
    ) -> tuple[MTState, MTState]:
        """Compute posterior and prior reconstructions for MTState."""
        rssm_model: M3RSSM = model  # type: ignore[assignment]
        posterior, _ = rssm_model.rollout_representation(
            actions=action_input,
            observations=observation_input,
            prev_state=rssm_model.initial_state(
                (vision_obs_input[:, 0], left_tactile_obs_input[:, 0], right_tactile_obs_input[:, 0])
            ),
        )
        prior = rssm_model.rollout_transition(
            actions=action_input[:, self.query_length :],
            prev_state=posterior[:, self.query_length - 1],
        )
        prior = cat_mtstates([posterior[:, : self.query_length], prior], dim=1)
        return posterior, prior

    @staticmethod
    def _denormalize_reconstructions(
        model: LightningModule,
        prior: MTState,
        posterior: MTState,
        observation_info: dict[str, torch.Tensor | bool],
    ) -> dict[str, torch.Tensor]:
        """Denormalize reconstructions for visualization with MTState."""
        vision_obs_target = torch.as_tensor(observation_info["vision_target"])
        left_tactile_obs_target = torch.as_tensor(observation_info["left_tactile_target"])
        right_tactile_obs_target = torch.as_tensor(observation_info["right_tactile_target"])
        vision_missing = bool(observation_info["vision_missing"])
        left_tactile_missing = bool(observation_info["left_tactile_missing"])
        right_tactile_missing = bool(observation_info["right_tactile_missing"])

        decoder_model = model  # type: ignore[assignment]
        posterior_vision_recon = decoder_model.vision_decoder.forward(posterior.feature)  # type: ignore[attr-defined, union-attr]
        posterior_left_tactile_recon = decoder_model.left_tactile_decoder.forward(posterior.feature)  # type: ignore[attr-defined, union-attr]
        posterior_right_tactile_recon = decoder_model.right_tactile_decoder.forward(posterior.feature)  # type: ignore[attr-defined, union-attr]
        prior_vision_recon = decoder_model.vision_decoder.forward(prior.feature)  # type: ignore[attr-defined, union-attr]
        prior_left_tactile_recon = decoder_model.left_tactile_decoder.forward(prior.feature)  # type: ignore[attr-defined, union-attr]
        prior_right_tactile_recon = decoder_model.right_tactile_decoder.forward(prior.feature)  # type: ignore[attr-defined, union-attr]

        prior_vision_recon = (prior_vision_recon + 1.0) / 2.0
        vision_obs_denorm = (vision_obs_target + 1.0) / 2.0
        posterior_vision_recon = (posterior_vision_recon + 1.0) / 2.0
        prior_left_tactile_recon = (prior_left_tactile_recon + 1.0) / 2.0
        left_tactile_obs_denorm = (left_tactile_obs_target + 1.0) / 2.0
        posterior_left_tactile_recon = (posterior_left_tactile_recon + 1.0) / 2.0
        prior_right_tactile_recon = (prior_right_tactile_recon + 1.0) / 2.0
        right_tactile_obs_denorm = (right_tactile_obs_target + 1.0) / 2.0
        posterior_right_tactile_recon = (posterior_right_tactile_recon + 1.0) / 2.0

        if vision_missing:
            vision_obs_denorm = torch.zeros_like(vision_obs_denorm)
        if left_tactile_missing:
            left_tactile_obs_denorm = torch.zeros_like(left_tactile_obs_denorm)
        if right_tactile_missing:
            right_tactile_obs_denorm = torch.zeros_like(right_tactile_obs_denorm)

        return {
            "prior_vision": prior_vision_recon,
            "observation_vision": vision_obs_denorm,
            "posterior_vision": posterior_vision_recon,
            "prior_left_tactile": prior_left_tactile_recon,
            "observation_left_tactile": left_tactile_obs_denorm,
            "posterior_left_tactile": posterior_left_tactile_recon,
            "prior_right_tactile": prior_right_tactile_recon,
            "observation_right_tactile": right_tactile_obs_denorm,
            "posterior_right_tactile": posterior_right_tactile_recon,
        }
