"""Callbacks for M3RSSM."""

import torch
from lightning import LightningModule

from models.callback import denormalize_tensor
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
            prev_state=rssm_model.initial_state((
                vision_obs_input[:, 0],
                left_tactile_obs_input[:, 0],
                right_tactile_obs_input[:, 0],
            )),
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
        left_tactile_init = torch.as_tensor(observation_info["left_tactile_init"])
        right_tactile_init = torch.as_tensor(observation_info["right_tactile_init"])
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

        vision_obs_denorm = denormalize_tensor(vision_obs_target)
        left_tactile_obs_denorm = LogMultimodalMRSSMOutput._restore_tactile_from_diff(
            left_tactile_obs_target,
            left_tactile_init,
        )
        right_tactile_obs_denorm = LogMultimodalMRSSMOutput._restore_tactile_from_diff(
            right_tactile_obs_target,
            right_tactile_init,
        )

        if vision_missing:
            vision_obs_denorm = torch.zeros_like(vision_obs_denorm)
        if left_tactile_missing:
            left_tactile_obs_denorm = torch.zeros_like(left_tactile_obs_denorm)
        if right_tactile_missing:
            right_tactile_obs_denorm = torch.zeros_like(right_tactile_obs_denorm)

        # Tactile diff observation: use raw diff (before processing) for visualization
        left_tactile_diff_raw = torch.as_tensor(observation_info["left_tactile_diff_raw"])
        right_tactile_diff_raw = torch.as_tensor(observation_info["right_tactile_diff_raw"])
        left_tactile_diff_obs = LogMultimodalMRSSMOutput._raw_diff_to_vis(left_tactile_diff_raw)
        right_tactile_diff_obs = LogMultimodalMRSSMOutput._raw_diff_to_vis(right_tactile_diff_raw)
        if left_tactile_missing:
            left_tactile_diff_obs = torch.zeros_like(left_tactile_diff_obs)
        if right_tactile_missing:
            right_tactile_diff_obs = torch.zeros_like(right_tactile_diff_obs)

        return {
            "prior_vision": denormalize_tensor(prior_vision_recon),
            "observation_vision": vision_obs_denorm,
            "posterior_vision": denormalize_tensor(posterior_vision_recon),
            "prior_left_tactile_diff": LogMultimodalMRSSMOutput._diff_norm_to_vis(prior_left_tactile_recon),
            "observation_left_tactile_diff": left_tactile_diff_obs,
            "posterior_left_tactile_diff": LogMultimodalMRSSMOutput._diff_norm_to_vis(posterior_left_tactile_recon),
            "prior_left_tactile": LogMultimodalMRSSMOutput._restore_tactile_from_diff(
                prior_left_tactile_recon,
                left_tactile_init,
            ),
            "observation_left_tactile": left_tactile_obs_denorm,
            "posterior_left_tactile": LogMultimodalMRSSMOutput._restore_tactile_from_diff(
                posterior_left_tactile_recon,
                left_tactile_init,
            ),
            "prior_right_tactile_diff": LogMultimodalMRSSMOutput._diff_norm_to_vis(prior_right_tactile_recon),
            "observation_right_tactile_diff": right_tactile_diff_obs,
            "posterior_right_tactile_diff": LogMultimodalMRSSMOutput._diff_norm_to_vis(posterior_right_tactile_recon),
            "prior_right_tactile": LogMultimodalMRSSMOutput._restore_tactile_from_diff(
                prior_right_tactile_recon,
                right_tactile_init,
            ),
            "observation_right_tactile": right_tactile_obs_denorm,
            "posterior_right_tactile": LogMultimodalMRSSMOutput._restore_tactile_from_diff(
                posterior_right_tactile_recon,
                right_tactile_init,
            ),
        }
