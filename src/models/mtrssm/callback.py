"""Callbacks for MTRSSM (vision-only single modality)."""

import torch
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor

from models.callback import (
    BaseLogRSSMOutput,
    add_timestep_labels,
    create_combined_video,
    log_video,
)
from models.mtrssm.core import MTRSSM
from models.state import State, cat_states


class LogMTRSSMOutput(BaseLogRSSMOutput):
    """Log MTRSSM output (vision-only single modality)."""

    def __init__(
        self,
        *,
        every_n_epochs: int,
        indices: list[int],
        query_length: int,
        fps: float,
    ) -> None:
        """Initialize LogMTRSSMOutput.

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
        )

    def _is_valid_model(self, pl_module: LightningModule) -> bool:
        """Check if the module is a valid MTRSSM model.

        Args:
            pl_module: PyTorch Lightning module

        Returns
        -------
        bool: True if valid, False otherwise
        """
        return isinstance(pl_module, MTRSSM)

    @staticmethod
    def _extract_episodes_from_batch(
        batch: tuple[Tensor, ...],
        device: torch.device,
    ) -> list[tuple[Tensor, ...]]:
        """Extract episodes from a batch for MTRSSM (vision-only).

        Args:
            batch: Batch tuple from dataloader
            device: Device to move tensors to

        Returns
        -------
        list: List of episode tuples
        """
        (
            action_input,
            vision_obs_input,
            _,  # action_target (not used)
            vision_obs_target,
        ) = (tensor.to(device) for tensor in batch)
        batch_size = action_input.shape[0]
        return [
            (
                action_input[i : i + 1],
                vision_obs_input[i : i + 1],
                vision_obs_target[i : i + 1],
            )
            for i in range(batch_size)
        ]

    def _process_episode(
        self,
        episode: tuple[Tensor, ...],
        model: LightningModule,
        stage: str,
        episode_idx: int,
        logger: WandbLogger,
    ) -> None:
        """Process a single episode and log the output for MTRSSM."""
        if not isinstance(model, MTRSSM):
            return

        (
            action_input,
            vision_obs_input,
            vision_obs_target,
        ) = episode

        # Compute posterior and prior reconstructions
        posterior, prior = self._compute_reconstructions(
            model,
            action_input,
            vision_obs_input,
        )

        # Denormalize for visualization
        video_data = self._denormalize_reconstructions(
            model,
            prior,
            posterior,
            vision_obs_target,
        )

        # Create and log combined vision video
        combined_video = create_combined_video(
            video_data["prior_vision"],
            video_data["observation_vision"],
            video_data["posterior_vision"],
        )
        combined_video_with_labels = add_timestep_labels(combined_video)

        key = f"{stage}/episode_{episode_idx}"
        log_video(combined_video_with_labels, key, logger, self.fps)

    def _compute_reconstructions(
        self,
        model: MTRSSM,
        action_input: Tensor,
        vision_obs_input: Tensor,
    ) -> tuple[State, State]:
        """Compute posterior and prior reconstructions.

        Args:
            model: MTRSSM model instance
            action_input: Action input tensor
            vision_obs_input: Vision observation input tensor

        Returns
        -------
        tuple[State, State]: (posterior, prior) states
        """
        posterior, _ = model.rollout_representation(
            actions=action_input,
            observations=vision_obs_input,
            prev_state=model.initial_state(vision_obs_input[:, 0]),
        )

        prior = model.rollout_transition(
            actions=action_input[:, self.query_length :],
            prev_state=posterior[:, self.query_length - 1],
        )
        prior = cat_states([posterior[:, : self.query_length], prior], dim=1)
        return posterior, prior

    @staticmethod
    def _denormalize_reconstructions(
        model: MTRSSM,
        prior: State,
        posterior: State,
        vision_obs_target: Tensor,
    ) -> dict[str, Tensor]:
        """Denormalize reconstructions for visualization.

        Args:
            model: MTRSSM model instance
            prior: Prior state
            posterior: Posterior state
            vision_obs_target: Vision observation target

        Returns
        -------
        dict[str, Tensor]: Dictionary with denormalized video data
        """
        # Compute reconstructions
        posterior_vision_recon = model.vision_decoder(posterior.feature)
        prior_vision_recon = model.vision_decoder(prior.feature)

        # Denormalize: from [-1, 1] to [0, 1]
        prior_vision_recon = (prior_vision_recon + 1.0) / 2.0
        vision_obs_denorm = (vision_obs_target + 1.0) / 2.0
        posterior_vision_recon = (posterior_vision_recon + 1.0) / 2.0

        return {
            "prior_vision": prior_vision_recon,
            "observation_vision": vision_obs_denorm,
            "posterior_vision": posterior_vision_recon,
        }
