"""M_MTRSSM: Multimodal MTRSSM with early fusion (3 modalities)."""

import torch
from torch import Tensor, nn

from models.m3rssm.state import MTState, stack_mtstates
from models.mtrssm.core import MTRSSM
from models.objective import likelihood


class M_MTRSSM(MTRSSM):  # noqa: N801
    """Multimodal MTRSSM with early fusion for 3 modalities (vision, left_tactile, right_tactile).

    Extends MTRSSM by replacing the single vision encoder with 3 encoders whose
    embeddings are concatenated and reduced via a fusion MLP (early fusion).
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        tactile_encoder: nn.Module,
        fusion_layer: nn.Module,
        left_tactile_decoder: nn.Module,
        right_tactile_decoder: nn.Module,
        tactile_recon_weight: float = 1.0,
        vision_recon_weight: float = 1.0,
        **kwargs: object,
    ) -> None:
        """Initialize M_MTRSSM.

        Args:
            tactile_encoder: Shared encoder for left and right tactile modalities.
            fusion_layer: MLP to reduce concatenated embeddings (3*embed_size → obs_embed_size).
            left_tactile_decoder: Decoder for left tactile modality.
            right_tactile_decoder: Decoder for right tactile modality.
            tactile_recon_weight: Weight for tactile reconstruction loss.
            vision_recon_weight: Weight for vision reconstruction loss.
            **kwargs: Passed to MTRSSM.__init__.
        """
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.tactile_encoder = tactile_encoder
        self.fusion_layer = fusion_layer
        self.left_tactile_decoder = left_tactile_decoder
        self.right_tactile_decoder = right_tactile_decoder
        self.tactile_recon_weight = tactile_recon_weight
        self.vision_recon_weight = vision_recon_weight

    def encode_observation(self, observation: tuple | Tensor) -> Tensor:  # type: ignore[override]
        """Encode 3-modality observations into a fused embedding.

        Args:
            observation: Tuple of (vision, left_tactile, right_tactile) tensors,
                each shape [*B, ...] or [*B, T, ...].

        Returns
        -------
        Tensor: Fused embedding of shape [*B, obs_embed_size] or [*B, T, obs_embed_size].
        """
        if isinstance(observation, tuple):
            vision_obs, left_tactile_obs, right_tactile_obs = observation
            vision_embed: Tensor = self.vision_encoder(vision_obs)  # type: ignore[no-any-return]
            left_tactile_embed: Tensor = self.tactile_encoder(left_tactile_obs)  # type: ignore[no-any-return]
            right_tactile_embed: Tensor = self.tactile_encoder(right_tactile_obs)  # type: ignore[no-any-return]
            concat_embed = torch.cat([vision_embed, left_tactile_embed, right_tactile_embed], dim=-1)
            return self.fusion_layer(concat_embed)  # type: ignore[no-any-return]
        return self.vision_encoder(observation)  # type: ignore[no-any-return]

    def initial_state(self, observation: tuple | Tensor) -> MTState:  # type: ignore[override]
        """Initialize hierarchical latent state from 3-modality observations.

        Returns
        -------
        MTState
            Initial hierarchical state constructed from the fused observation embedding.
        """
        obs_embed = self.encode_observation(observation)
        device = obs_embed.device

        h = self.init_proj(obs_embed)
        higher_h = h[..., : self.hd_dim]
        lower_h = h[..., self.hd_dim :]

        self.h_rnn.hidden = higher_h
        self.l_rnn.hidden = lower_h

        h_prior_logits = self.h_prior(higher_h)
        l_prior_logits = self.l_prior(lower_h)
        h_prior_dist = self.h_dist(h_prior_logits)
        l_prior_dist = self.l_dist(l_prior_logits)

        return MTState(
            deter_h=higher_h,
            deter_l=lower_h,
            distribution_h=h_prior_dist,
            distribution_l=l_prior_dist,
            hidden_h=higher_h,
            hidden_l=lower_h,
        ).to(device)

    def rollout_representation(  # noqa: PLR0914
        self,
        *,
        actions: Tensor,
        observations: tuple | Tensor,
        prev_state: MTState,
    ) -> tuple[MTState, MTState]:
        """Rollout representation with 3-modality early fusion.

        Args:
            actions: Action sequence. Shape: [B, T, action_size]
            observations: Tuple of (vision, left_tactile, right_tactile),
                each shape [B, T, ...].
            prev_state: Previous hierarchical state.

        Returns
        -------
        tuple[MTState, MTState]: (posterior, prior) states.

        Raises
        ------
        TypeError
            If observations is not a tuple.
        """
        if not isinstance(observations, tuple):
            msg = "M_MTRSSM requires tuple of (vision_obs, left_tactile_obs, right_tactile_obs)"
            raise TypeError(msg)

        vision_obs, left_tactile_obs, right_tactile_obs = observations

        vision_embed: Tensor = self.vision_encoder(vision_obs)  # type: ignore[no-any-return]
        left_tactile_embed: Tensor = self.tactile_encoder(left_tactile_obs)  # type: ignore[no-any-return]
        right_tactile_embed: Tensor = self.tactile_encoder(right_tactile_obs)  # type: ignore[no-any-return]

        concat_embed = torch.cat([vision_embed, left_tactile_embed, right_tactile_embed], dim=-1)
        fused_embed: Tensor = self.fusion_layer(concat_embed)

        self._set_prev_hiddens(fused_embed.shape[0], prev_state=prev_state)

        priors: list[MTState] = []
        posteriors: list[MTState] = []

        for t in range(fused_embed.shape[1]):
            prev_l_stoch = prev_state.stoch_l
            prev_h_stoch = prev_state.stoch_h
            prev_l_deter = prev_state.deter_l
            prev_h_deter = prev_state.deter_h

            l_deter, l_prior_dist = self._compute_lower_prior(
                actions[:, t],
                prev_l_stoch,
                prev_h_stoch,
                prev_l_deter,
            )

            fused_obs_embed = fused_embed[:, t]
            _, fused_logits = self._compute_lower_posterior_with_logits(
                fused_obs_embed,
                l_deter,
                self.vision_representation,
            )
            l_posterior_dist = self.l_dist(fused_logits)
            l_stoch = l_posterior_dist.rsample()

            h_deter, h_prior_dist, h_posterior_dist = self._compute_higher_prior_posterior(
                l_deter,
                prev_h_deter,
                prev_h_stoch,
            )
            h_stoch = h_posterior_dist.rsample()

            prior_state = MTState(
                deter_h=h_deter,
                deter_l=l_deter,
                distribution_h=h_prior_dist,
                distribution_l=l_prior_dist,
                hidden_h=self.h_rnn.hidden,  # type: ignore[arg-type]
                hidden_l=self.l_rnn.hidden,  # type: ignore[arg-type]
            )
            posterior_state = MTState(
                deter_h=h_deter,
                deter_l=l_deter,
                distribution_h=h_posterior_dist,
                distribution_l=l_posterior_dist,
                hidden_h=self.h_rnn.hidden,  # type: ignore[arg-type]
                hidden_l=self.l_rnn.hidden,  # type: ignore[arg-type]
                stoch_h=h_stoch,
                stoch_l=l_stoch,
            )
            priors.append(prior_state)
            posteriors.append(posterior_state)
            prev_state = posterior_state

        prior = stack_mtstates(priors, dim=1)
        posterior = stack_mtstates(posteriors, dim=1)
        return posterior, prior

    def decode_state(self, state: MTState) -> dict[str, Tensor]:  # type: ignore[override]
        """Decode hierarchical state into 3-modality reconstructions.

        Returns
        -------
        dict[str, Tensor]
            Dictionary of reconstructed observations for each modality.
        """
        feature = state.feature
        return {
            "recon/vision": self.vision_decoder(feature),
            "recon/left_tactile": self.left_tactile_decoder(feature),
            "recon/right_tactile": self.right_tactile_decoder(feature),
        }

    def compute_reconstruction_loss(
        self,
        reconstructions: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute weighted reconstruction losses for 3 modalities.

        Returns
        -------
        dict[str, Tensor]
            Dictionary of total and per-modality reconstruction losses.
        """
        vision_recon_loss = likelihood(
            prediction=reconstructions["recon/vision"],
            target=targets["recon/vision"],
            event_ndims=3,
        )
        left_tactile_recon_loss = likelihood(
            prediction=reconstructions["recon/left_tactile"],
            target=targets["recon/left_tactile"],
            event_ndims=3,
        )
        right_tactile_recon_loss = likelihood(
            prediction=reconstructions["recon/right_tactile"],
            target=targets["recon/right_tactile"],
            event_ndims=3,
        )
        w_vision = self.vision_recon_weight
        w_tactile = self.tactile_recon_weight
        return {
            "recon": (
                w_vision * vision_recon_loss
                + w_tactile * (left_tactile_recon_loss + right_tactile_recon_loss)
            ),
            "recon/vision": vision_recon_loss,
            "recon/left_tactile": left_tactile_recon_loss,
            "recon/right_tactile": right_tactile_recon_loss,
        }

    @staticmethod
    def get_observations_from_batch(
        batch: tuple[Tensor, ...],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Extract 3-modality observation sequences from batch.

        Args:
            batch: Batch tuple.
                Format: (action_input, vision_obs_input, left_tactile_obs_input,
                        right_tactile_obs_input, ...)

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]: (vision, left_tactile, right_tactile) inputs.
        """
        return batch[1], batch[2], batch[3]

    @staticmethod
    def get_initial_observation(
        observations: tuple[Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Extract t=0 frames from 3-modality observation sequences.

        Args:
            observations: Tuple of (vision, left_tactile, right_tactile), each [B, T, ...].

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]: Initial observations each shape [B, ...].
        """
        vision_obs, left_tactile_obs, right_tactile_obs = observations
        return vision_obs[:, 0], left_tactile_obs[:, 0], right_tactile_obs[:, 0]

    @staticmethod
    def get_targets_from_batch(batch: tuple[Tensor, ...]) -> dict[str, Tensor]:
        """Extract 3-modality reconstruction targets from batch.

        Args:
            batch: Batch tuple.
                Format: (action_input, vision_obs_input, left_tactile_obs_input,
                        right_tactile_obs_input, action_target, vision_obs_target,
                        left_tactile_obs_target, right_tactile_obs_target, ...)
                Indices 5..7 are the reconstruction targets.

        Returns
        -------
        dict[str, Tensor]: Reconstruction targets for each modality.
        """
        return {
            "recon/vision": batch[5],
            "recon/left_tactile": batch[6],
            "recon/right_tactile": batch[7],
        }
