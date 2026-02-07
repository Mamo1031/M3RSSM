"""MoPoE-MRSSM module for multimodal recurrent state-space model."""

import torch
from torch import Tensor, nn

from models.core import BaseRSSM
from models.networks import Representation, Transition
from models.objective import likelihood
from models.state import State, stack_states


class MoPoE_MRSSM(BaseRSSM):  # noqa: N801
    """Multimodal Recurrent State-Space Model with MoPoE-style posteriors (MoPoE-MRSSM).

    This model combines Product of Experts (PoE) and Mixture of Experts (MoE):
    1. Fused posteriors are created by PoE fusion of vision, left_tactile, and right_tactile posteriors
       - 4 PoE combinations: (V+L), (V+R), (L+R), (V+L+R)
    2. Mixed posterior is created by MoE fusion of 8 experts:
       {φ, V, L, R, (V+L), (V+R), (L+R), (V+L+R)} with equal weights (1/8 each)
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        vision_representation: Representation,
        left_tactile_representation: Representation,
        right_tactile_representation: Representation,
        transition: Transition,
        vision_encoder: nn.Module,
        tactile_encoder: nn.Module,
        vision_decoder: nn.Module,
        left_tactile_decoder: nn.Module,
        right_tactile_decoder: nn.Module,
        init_proj: nn.Module,
        kl_coeff: float,
        use_kl_balancing: bool,
    ) -> None:
        """Initialize MoPoE-MRSSM.

        Args:
            vision_representation: Representation network for vision modality.
            left_tactile_representation: Representation network for left tactile modality.
            right_tactile_representation: Representation network for right tactile modality.
            transition: Transition network (RSSM prior).
            vision_encoder: Vision encoder network.
            tactile_encoder: Tactile encoder network (shared for left and right).
            vision_decoder: Vision decoder network.
            left_tactile_decoder: Left tactile decoder network.
            right_tactile_decoder: Right tactile decoder network.
            init_proj: Initial projection network.
            kl_coeff: KL divergence coefficient.
            use_kl_balancing: Whether to use KL balancing.
        """
        super().__init__(
            representation=vision_representation,
            transition=transition,
            init_proj=init_proj,
            kl_coeff=kl_coeff,
            use_kl_balancing=use_kl_balancing,
        )
        self.vision_representation = vision_representation
        self.left_tactile_representation = left_tactile_representation
        self.right_tactile_representation = right_tactile_representation
        self.vision_encoder = vision_encoder
        self.tactile_encoder = tactile_encoder
        self.vision_decoder = vision_decoder
        self.left_tactile_decoder = left_tactile_decoder
        self.right_tactile_decoder = right_tactile_decoder

    @staticmethod
    def _compute_posterior_with_logits(
        obs_embed: Tensor,
        prior_state: State,
        representation: Representation,
    ) -> tuple[State, Tensor]:
        """Compute posterior state and return logits for fusion.

        Args:
            obs_embed: Observation embedding. Shape: [*B, obs_embed_size]
            prior_state: Prior state.
            representation: Representation network to use.

        Returns
        -------
        tuple[State, Tensor]
            (posterior_state, logits) where logits shape is [*B, class_size * category_size]
        """
        projector_input = torch.cat([prior_state.deter, obs_embed], -1)
        stoch_source = representation.rnn_to_post_projector(projector_input)
        distribution = representation.distribution_factory.forward(stoch_source)
        posterior = State(deter=prior_state.deter, distribution=distribution)
        return posterior, stoch_source

    def _poe_fusion_categorical_2mod(self, logits1: Tensor, logits2: Tensor, prior_state: State) -> State:
        """Fuse 2 categorical distributions using Product of Experts (PoE).

        PoE for categorical distributions is computed by summing log-probabilities.

        Args:
            logits1: First modality posterior logits. Shape: [*B, class_size * category_size]
            logits2: Second modality posterior logits. Shape: [*B, class_size * category_size]
            prior_state: Prior state (for deterministic part).

        Returns
        -------
        State: Fused posterior state.
        """
        # Convert each expert's logits to log-probabilities
        log_probs1 = torch.nn.functional.log_softmax(logits1, dim=-1)
        log_probs2 = torch.nn.functional.log_softmax(logits2, dim=-1)

        # Sum log-probabilities: log p_fused ∝ log p1 + log p2
        fused_log_probs = log_probs1 + log_probs2

        # Create categorical distribution
        fused_distribution = self.vision_representation.distribution_factory.forward(fused_log_probs)

        return State(deter=prior_state.deter, distribution=fused_distribution)

    def _poe_fusion_categorical_3mod(
        self, logits1: Tensor, logits2: Tensor, logits3: Tensor, prior_state: State,
    ) -> State:
        """Fuse 3 categorical distributions using Product of Experts (PoE).

        PoE for categorical distributions is computed by summing log-probabilities.

        Args:
            logits1: First modality posterior logits. Shape: [*B, class_size * category_size]
            logits2: Second modality posterior logits. Shape: [*B, class_size * category_size]
            logits3: Third modality posterior logits. Shape: [*B, class_size * category_size]
            prior_state: Prior state (for deterministic part).

        Returns
        -------
        State: Fused posterior state.
        """
        # Convert each expert's logits to log-probabilities
        log_probs1 = torch.nn.functional.log_softmax(logits1, dim=-1)
        log_probs2 = torch.nn.functional.log_softmax(logits2, dim=-1)
        log_probs3 = torch.nn.functional.log_softmax(logits3, dim=-1)

        # Sum log-probabilities: log p_fused ∝ log p1 + log p2 + log p3
        fused_log_probs = log_probs1 + log_probs2 + log_probs3

        # Create categorical distribution
        fused_distribution = self.vision_representation.distribution_factory.forward(fused_log_probs)

        return State(deter=prior_state.deter, distribution=fused_distribution)

    def _moe_fusion_categorical(
        self,
        prior_log_probs: Tensor,
        vision_log_probs: Tensor,
        left_tactile_log_probs: Tensor,
        right_tactile_log_probs: Tensor,
        vl_poe_log_probs: Tensor,
        vr_poe_log_probs: Tensor,
        lr_poe_log_probs: Tensor,
        vlr_poe_log_probs: Tensor,
        prior_state: State,
    ) -> State:
        """Fuse categorical distributions using Mixture of Experts (MoE).

        MoE fusion is performed by weighted averaging of probability distributions
        from all 8 experts {φ, V, L, R, (V+L), (V+R), (L+R), (V+L+R)} with equal weights (1/8 each).

        Args:
            prior_log_probs: Prior (φ) log-probabilities. Shape: [*B, class_size * category_size]
            vision_log_probs: Vision (V) log-probabilities. Shape: [*B, class_size * category_size]
            left_tactile_log_probs: Left tactile (L) log-probabilities. Shape: [*B, class_size * category_size]
            right_tactile_log_probs: Right tactile (R) log-probabilities. Shape: [*B, class_size * category_size]
            vl_poe_log_probs: (V+L) PoE fused log-probabilities. Shape: [*B, class_size * category_size]
            vr_poe_log_probs: (V+R) PoE fused log-probabilities. Shape: [*B, class_size * category_size]
            lr_poe_log_probs: (L+R) PoE fused log-probabilities. Shape: [*B, class_size * category_size]
            vlr_poe_log_probs: (V+L+R) PoE fused log-probabilities. Shape: [*B, class_size * category_size]
            prior_state: Prior state (for deterministic part).

        Returns
        -------
        State: Mixed posterior state.
        """
        # Weighted average in log-probability space for numerical stability:
        weight = 1.0 / 8.0
        log_weight = torch.log(torch.tensor(weight, device=prior_log_probs.device, dtype=prior_log_probs.dtype))

        # Stack log-probabilities with log-weight: [*B, 8, class_size * category_size]
        weighted_log_probs = torch.stack(
            [
                log_weight + prior_log_probs,
                log_weight + vision_log_probs,
                log_weight + left_tactile_log_probs,
                log_weight + right_tactile_log_probs,
                log_weight + vl_poe_log_probs,
                log_weight + vr_poe_log_probs,
                log_weight + lr_poe_log_probs,
                log_weight + vlr_poe_log_probs,
            ],
            dim=-2,
        )  # [*B, 8, class_size * category_size]

        # Log-sum-exp for numerical stability: log(sum(exp(x))) = logsumexp(x)
        mixed_log_probs = torch.logsumexp(weighted_log_probs, dim=-2)  # [*B, class_size * category_size]

        # Ensure mixed_log_probs is 2D: [B, class_size * category_size]
        min_dim_for_2d = 2
        if mixed_log_probs.dim() > min_dim_for_2d:
            mixed_log_probs = mixed_log_probs.view(mixed_log_probs.shape[0], -1)

        mixed_distribution = self.vision_representation.distribution_factory.forward(mixed_log_probs)

        return State(deter=prior_state.deter, distribution=mixed_distribution)

    def encode_observation(self, observation: tuple[Tensor, Tensor, Tensor] | Tensor) -> Tensor:  # type: ignore[override]
        """Encode observation(s) into fused embedding.

        Args:
            observation: Observation(s) to encode.
                For tuple: (vision_obs, left_tactile_obs, right_tactile_obs) each shape: [*B, T, ...] or [*B, ...]
                For Tensor: single observation (for backward compatibility)

        Returns
        -------
        Tensor: Observation embedding. Shape: [*B, T, obs_embed_size] or [*B, obs_embed_size]
        """
        if isinstance(observation, tuple):
            vision_obs, left_tactile_obs, right_tactile_obs = observation
            vision_embed: Tensor = self.vision_encoder(vision_obs)  # type: ignore[no-any-return]
            left_tactile_embed: Tensor = self.tactile_encoder(left_tactile_obs)  # type: ignore[no-any-return]
            right_tactile_embed: Tensor = self.tactile_encoder(right_tactile_obs)  # type: ignore[no-any-return]
            return (vision_embed + left_tactile_embed + right_tactile_embed) / 3.0
        return observation

    def rollout_representation(  # noqa: PLR0914
        self,
        *,
        actions: Tensor,
        observations: Tensor | tuple[Tensor, ...],
        prev_state: State,
    ) -> tuple[State, State]:
        """Rollout representation and compute posteriors with PoE and MoE fusion.

        Args:
            actions: The actions to rollout the representation. Shape: [B, T, action_size]
            observations: The observations to rollout the representation.
                For tuple: (vision_obs, left_tactile_obs, right_tactile_obs) each shape: [B, T, ...]
            prev_state: The previous state to rollout the representation.

        Returns
        -------
        tuple[State, State]
            (mixed posterior, prior) states.

        Raises
        ------
        TypeError
            If observations is not a tuple.
        """
        if not isinstance(observations, tuple):
            msg = "MoPoE-MRSSM requires tuple of (vision_obs, left_tactile_obs, right_tactile_obs)"
            raise TypeError(msg)

        vision_obs, left_tactile_obs, right_tactile_obs = observations

        vision_embed: Tensor = self.vision_encoder(vision_obs)  # type: ignore[no-any-return]
        left_tactile_embed: Tensor = self.tactile_encoder(left_tactile_obs)  # type: ignore[no-any-return]
        right_tactile_embed: Tensor = self.tactile_encoder(right_tactile_obs)  # type: ignore[no-any-return]

        priors: list[State] = []
        mixed_posteriors: list[State] = []

        for t in range(vision_embed.shape[1]):
            prior = self.transition.forward(actions[:, t], prev_state)

            # Get prior log-probabilities (φ)
            prior_logits = self.transition.rnn_to_prior_projector(prior.deter)
            prior_log_probs = torch.nn.functional.log_softmax(prior_logits, dim=-1)

            # Vision-only posterior q_V (with logits for fusion)
            vision_obs_embed = vision_embed[:, t]
            _, vision_logits = self._compute_posterior_with_logits(
                vision_obs_embed,
                prior,
                self.vision_representation,
            )
            vision_log_probs = torch.nn.functional.log_softmax(vision_logits, dim=-1)

            # Left tactile-only posterior q_L (with logits for fusion)
            left_tactile_obs_embed = left_tactile_embed[:, t]
            _, left_tactile_logits = self._compute_posterior_with_logits(
                left_tactile_obs_embed,
                prior,
                self.left_tactile_representation,
            )
            left_tactile_log_probs = torch.nn.functional.log_softmax(left_tactile_logits, dim=-1)

            # Right tactile-only posterior q_R (with logits for fusion)
            right_tactile_obs_embed = right_tactile_embed[:, t]
            _, right_tactile_logits = self._compute_posterior_with_logits(
                right_tactile_obs_embed,
                prior,
                self.right_tactile_representation,
            )
            right_tactile_log_probs = torch.nn.functional.log_softmax(right_tactile_logits, dim=-1)

            # Compute 4 PoE fusions: (V+L), (V+R), (L+R), (V+L+R)
            vl_poe_log_probs = vision_log_probs + left_tactile_log_probs
            vr_poe_log_probs = vision_log_probs + right_tactile_log_probs
            lr_poe_log_probs = left_tactile_log_probs + right_tactile_log_probs
            vlr_poe_log_probs = vision_log_probs + left_tactile_log_probs + right_tactile_log_probs

            # Create mixed posterior by MoE fusion of {φ, V, L, R, (V+L), (V+R), (L+R), (V+L+R)}
            mixed_posterior = self._moe_fusion_categorical(
                prior_log_probs,
                vision_log_probs,
                left_tactile_log_probs,
                right_tactile_log_probs,
                vl_poe_log_probs,
                vr_poe_log_probs,
                lr_poe_log_probs,
                vlr_poe_log_probs,
                prior,
            )

            priors.append(prior)
            mixed_posteriors.append(mixed_posterior)

            prev_state = mixed_posterior

        prior = stack_states(priors, dim=1)
        posterior_mixed = stack_states(mixed_posteriors, dim=1)
        return posterior_mixed, prior

    def decode_state(self, state: State) -> dict[str, Tensor]:
        """Decode mixed state into reconstructions for each modality.

        Args:
            state: The state to decode.

        Returns
        -------
        dict[str, Tensor]: Dictionary with reconstructions for "recon/vision", "recon/left_tactile", and "recon/right_tactile".
        """
        vision_recon = self.vision_decoder(state.feature)
        left_tactile_recon = self.left_tactile_decoder(state.feature)
        right_tactile_recon = self.right_tactile_decoder(state.feature)
        return {
            "recon/vision": vision_recon,
            "recon/left_tactile": left_tactile_recon,
            "recon/right_tactile": right_tactile_recon,
        }

    @staticmethod
    def compute_reconstruction_loss(
        reconstructions: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute reconstruction losses.

        Args:
            reconstructions: Dictionary with "recon/vision", "recon/left_tactile", and "recon/right_tactile" keys.
            targets: Dictionary with "recon/vision", "recon/left_tactile", and "recon/right_tactile" keys.

        Returns
        -------
        dict[str, Tensor]: Dictionary with "recon", "recon/vision", "recon/left_tactile", and "recon/right_tactile" keys.
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
        return {
            "recon": vision_recon_loss + left_tactile_recon_loss + right_tactile_recon_loss,
            "recon/vision": vision_recon_loss,
            "recon/left_tactile": left_tactile_recon_loss,
            "recon/right_tactile": right_tactile_recon_loss,
        }

    @staticmethod
    def get_observations_from_batch(batch: tuple[Tensor, ...]) -> tuple[Tensor, Tensor, Tensor]:
        """Extract observation sequences from batch.

        Args:
            batch: Batch tuple.
                Format: (action_input, vision_obs_input, left_tactile_obs_input, right_tactile_obs_input,
                        action_target, vision_obs_target, left_tactile_obs_target, right_tactile_obs_target)

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]: (vision_obs_input, left_tactile_obs_input, right_tactile_obs_input) each shape: [B, T, ...]
        """
        return batch[1], batch[2], batch[3]

    @staticmethod
    def get_initial_observation(observations: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Extract initial observations from observation sequences.

        Args:
            observations: Tuple of (vision_obs, left_tactile_obs, right_tactile_obs) each shape: [B, T, ...]

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]: Initial observations each shape: [B, ...]
        """
        vision_obs, left_tactile_obs, right_tactile_obs = observations
        return vision_obs[:, 0], left_tactile_obs[:, 0], right_tactile_obs[:, 0]

    @staticmethod
    def get_targets_from_batch(batch: tuple[Tensor, ...]) -> dict[str, Tensor]:
        """Extract reconstruction targets from batch.

        Args:
            batch: Batch tuple.
                Format: (action_input, vision_obs_input, left_tactile_obs_input, right_tactile_obs_input,
                        action_target, vision_obs_target, left_tactile_obs_target, right_tactile_obs_target)

        Returns
        -------
        dict[str, Tensor]: Dictionary with "recon/vision", "recon/left_tactile", and "recon/right_tactile" keys.
        """
        return {
            "recon/vision": batch[5],
            "recon/left_tactile": batch[6],
            "recon/right_tactile": batch[7],
        }
