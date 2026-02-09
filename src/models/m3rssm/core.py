"""M3RSSM module for multimodal temporal recurrent state-space model."""

import torch
from distribution_extension import Distribution, MultiOneHotFactory, kl_divergence
from torch import Tensor, nn

from models.m3rssm.state import MTState, stack_mtstates
from models.mopoe_mrssm.core import MoPoE_MRSSM
from models.networks import Representation, Transition


class MTRNN(nn.Module):
    """Multi-Timescale RNN cell."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        bias: bool = True,
        tau: float = 2.0,
    ) -> None:
        """Initialize MTRNN."""
        super().__init__()
        if tau <= 1.0:
            msg = "tau must be greater than 1.0"
            raise ValueError(msg)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.tau = tau
        self._d2h = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self._input2h = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.hidden: Tensor | None = None

    def _compute_mtrnn(self, inputs: Tensor, prev_d: Tensor) -> Tensor:
        if self.hidden is None:
            self.hidden = torch.zeros(
                inputs.shape[0],
                self.hidden_dim,
                device=inputs.device,
                dtype=inputs.dtype,
            )
        self.hidden = (1 - 1 / self.tau) * self.hidden + (self._d2h(prev_d) + self._input2h(inputs)) / self.tau
        return torch.tanh(self.hidden)

    def forward(self, inputs: Tensor, prev_d: Tensor) -> Tensor:
        """Forward pass of MTRNN."""
        return self._compute_mtrnn(inputs, prev_d)


class M3RSSM(MoPoE_MRSSM):
    """Multimodal Temporal RSSM with MoPoE fusion (vision + left/right tactile + action)."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        vision_representation: Representation,
        left_tactile_representation: Representation,
        right_tactile_representation: Representation,
        vision_encoder: nn.Module,
        tactile_encoder: nn.Module,
        vision_decoder: nn.Module,
        left_tactile_decoder: nn.Module,
        right_tactile_decoder: nn.Module,
        init_proj: nn.Module,
        kl_coeff: float,
        use_kl_balancing: bool,
        tactile_recon_weight: float = 1.0,
        vision_recon_weight: float = 1.0,
        # MTRSSM specific parameters
        action_size: int,
        hd_dim: int,
        hs_dim: int,
        ld_dim: int,
        ls_dim: int,
        l_tau: float,
        h_tau: float,
        l_prior: nn.Module,
        l_posterior: nn.Module,
        h_prior: nn.Module,
        h_posterior: nn.Module,
        l_dist: MultiOneHotFactory,
        h_dist: MultiOneHotFactory,
        w_kl_h: float = 1.0,
    ) -> None:
        """Initialize M3RSSM."""
        dummy_transition = Transition(
            deterministic_size=ld_dim,
            hidden_size=ld_dim,
            action_size=1,
            distribution_config=[1, 1],
            activation_name="ELU",
        )
        super().__init__(
            vision_representation=vision_representation,
            left_tactile_representation=left_tactile_representation,
            right_tactile_representation=right_tactile_representation,
            transition=dummy_transition,
            vision_encoder=vision_encoder,
            tactile_encoder=tactile_encoder,
            vision_decoder=vision_decoder,
            left_tactile_decoder=left_tactile_decoder,
            right_tactile_decoder=right_tactile_decoder,
            init_proj=init_proj,
            kl_coeff=kl_coeff,
            use_kl_balancing=use_kl_balancing,
            tactile_recon_weight=tactile_recon_weight,
            vision_recon_weight=vision_recon_weight,
        )

        self.action_dim = action_size
        self.hd_dim = hd_dim
        self.hs_dim = hs_dim
        self.ld_dim = ld_dim
        self.ls_dim = ls_dim
        self.w_kl_h = w_kl_h

        self.l_rnn = MTRNN(
            input_dim=action_size + ls_dim + hs_dim,
            hidden_dim=ld_dim,
            tau=l_tau,
        )
        self.h_rnn = MTRNN(
            input_dim=hs_dim,
            hidden_dim=hd_dim,
            tau=h_tau,
        )

        self.l_prior = l_prior
        self.l_posterior = l_posterior
        self.h_prior = h_prior
        self.h_posterior = h_posterior
        self.l_dist = l_dist
        self.h_dist = h_dist

    @property
    def feature_dim(self) -> int:
        """Feature dimension (hd_dim + hs_dim + ld_dim + ls_dim)."""
        return self.hd_dim + self.hs_dim + self.ld_dim + self.ls_dim

    def _set_prev_hiddens(
        self,
        batch_size: int,
        *,
        init_obs: Tensor | None = None,
        prev_state: MTState | None = None,
        device: torch.device | None = None,
    ) -> None:
        if device is None:
            device = next(self.parameters()).device
        if prev_state is not None:
            self.l_rnn.hidden = prev_state.hidden_l
            self.h_rnn.hidden = prev_state.hidden_h
            return
        if init_obs is not None:
            h = self.init_proj(init_obs)
            higher_h = h[..., : self.hd_dim]
            lower_h = h[..., self.hd_dim :]
            self.h_rnn.hidden = higher_h
            self.l_rnn.hidden = lower_h
            return
        self.h_rnn.hidden = torch.zeros(batch_size, self.hd_dim, device=device)
        self.l_rnn.hidden = torch.zeros(batch_size, self.ld_dim, device=device)

    def _compute_lower_posterior_with_logits(
        self,
        obs_embed: Tensor,
        prior_l_deter: Tensor,
        representation: Representation,
    ) -> tuple[Tensor, Tensor]:
        projector_input = torch.cat([prior_l_deter, obs_embed], -1)
        stoch_source = representation.rnn_to_post_projector(projector_input)
        return stoch_source, stoch_source

    def _compute_lower_prior(
        self,
        action: Tensor,
        prev_l_stoch: Tensor,
        prev_h_stoch: Tensor,
        prev_l_deter: Tensor,
    ) -> tuple[Tensor, Distribution]:
        prev_l_stoch_flat = prev_l_stoch.flatten(start_dim=1) if prev_l_stoch.dim() > 2 else prev_l_stoch
        prev_h_stoch_flat = prev_h_stoch.flatten(start_dim=1) if prev_h_stoch.dim() > 2 else prev_h_stoch
        l_input = torch.cat([action, prev_l_stoch_flat, prev_h_stoch_flat], dim=-1)
        l_deter = self.l_rnn(l_input, prev_l_deter)
        l_prior_logits = self.l_prior(l_deter)
        l_prior_dist = self.l_dist(l_prior_logits)
        return l_deter, l_prior_dist

    def _compute_higher_prior_posterior(
        self,
        l_deter: Tensor,
        prev_h_deter: Tensor,
        prev_h_stoch: Tensor,
    ) -> tuple[Tensor, Distribution, Distribution]:
        prev_h_stoch_flat = prev_h_stoch.flatten(start_dim=1) if prev_h_stoch.dim() > 2 else prev_h_stoch
        h_deter = self.h_rnn(prev_h_stoch_flat, prev_h_deter)
        h_prior_logits = self.h_prior(h_deter)
        h_prior_dist = self.h_dist(h_prior_logits)
        h_posterior_input = torch.cat([l_deter, h_deter], dim=-1)
        h_posterior_logits = self.h_posterior(h_posterior_input)
        h_posterior_dist = self.h_dist(h_posterior_logits)
        return h_deter, h_prior_dist, h_posterior_dist

    def initial_state(self, observation: tuple[Tensor, Tensor, Tensor] | Tensor) -> MTState:  # type: ignore[override]
        """Initialize hierarchical latent state."""
        obs_embed = self.encode_observation(observation) if isinstance(observation, tuple) else observation
        batch_size = obs_embed.shape[0]
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
        observations: Tensor | tuple[Tensor, ...],
        prev_state: MTState,
    ) -> tuple[MTState, MTState]:
        """Rollout representation with MoPoE fusion on hierarchical states."""
        if not isinstance(observations, tuple):
            msg = "M3RSSM requires tuple of (vision_obs, left_tactile_obs, right_tactile_obs)"
            raise TypeError(msg)

        vision_obs, left_tactile_obs, right_tactile_obs = observations
        vision_embed: Tensor = self.vision_encoder(vision_obs)  # type: ignore[no-any-return]
        left_tactile_embed: Tensor = self.tactile_encoder(left_tactile_obs)  # type: ignore[no-any-return]
        right_tactile_embed: Tensor = self.tactile_encoder(right_tactile_obs)  # type: ignore[no-any-return]

        self._set_prev_hiddens(vision_embed.shape[0], prev_state=prev_state)

        priors: list[MTState] = []
        mixed_posteriors: list[MTState] = []

        for t in range(vision_embed.shape[1]):
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

            l_prior_logits = self.l_prior(l_deter)
            prior_log_probs = torch.nn.functional.log_softmax(l_prior_logits, dim=-1)

            vision_obs_embed = vision_embed[:, t]
            _, vision_logits = self._compute_lower_posterior_with_logits(
                vision_obs_embed,
                l_deter,
                self.vision_representation,
            )
            vision_log_probs = torch.nn.functional.log_softmax(vision_logits, dim=-1)

            left_tactile_obs_embed = left_tactile_embed[:, t]
            _, left_logits = self._compute_lower_posterior_with_logits(
                left_tactile_obs_embed,
                l_deter,
                self.left_tactile_representation,
            )
            left_log_probs = torch.nn.functional.log_softmax(left_logits, dim=-1)

            right_tactile_obs_embed = right_tactile_embed[:, t]
            _, right_logits = self._compute_lower_posterior_with_logits(
                right_tactile_obs_embed,
                l_deter,
                self.right_tactile_representation,
            )
            right_log_probs = torch.nn.functional.log_softmax(right_logits, dim=-1)

            vl_poe_log_probs = vision_log_probs + left_log_probs
            vr_poe_log_probs = vision_log_probs + right_log_probs
            lr_poe_log_probs = left_log_probs + right_log_probs
            vlr_poe_log_probs = vision_log_probs + left_log_probs + right_log_probs

            weight = 1.0 / 8.0
            log_weight = torch.log(torch.tensor(weight, device=prior_log_probs.device, dtype=prior_log_probs.dtype))
            weighted_log_probs = torch.stack(
                [
                    log_weight + prior_log_probs,
                    log_weight + vision_log_probs,
                    log_weight + left_log_probs,
                    log_weight + right_log_probs,
                    log_weight + vl_poe_log_probs,
                    log_weight + vr_poe_log_probs,
                    log_weight + lr_poe_log_probs,
                    log_weight + vlr_poe_log_probs,
                ],
                dim=-2,
            )
            mixed_log_probs = torch.logsumexp(weighted_log_probs, dim=-2)
            if mixed_log_probs.dim() > 2:
                mixed_log_probs = mixed_log_probs.view(mixed_log_probs.shape[0], -1)

            l_posterior_dist = self.l_dist(mixed_log_probs)
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
                hidden_h=self.h_rnn.hidden,
                hidden_l=self.l_rnn.hidden,
            )
            posterior_state = MTState(
                deter_h=h_deter,
                deter_l=l_deter,
                distribution_h=h_posterior_dist,
                distribution_l=l_posterior_dist,
                hidden_h=self.h_rnn.hidden,
                hidden_l=self.l_rnn.hidden,
                stoch_h=h_stoch,
                stoch_l=l_stoch,
            )
            priors.append(prior_state)
            mixed_posteriors.append(posterior_state)
            prev_state = posterior_state

        prior = stack_mtstates(priors, dim=1)
        posterior_mixed = stack_mtstates(mixed_posteriors, dim=1)
        return posterior_mixed, prior

    def rollout_transition(self, *, actions: Tensor, prev_state: MTState) -> MTState:  # type: ignore[override]
        """Rollout the transition model using 2-layer RNN."""
        self._set_prev_hiddens(actions.shape[0], prev_state=prev_state)
        priors: list[MTState] = []

        for t in range(actions.shape[1]):
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
            prev_h_stoch_flat = prev_h_stoch.flatten(start_dim=1) if prev_h_stoch.dim() > 2 else prev_h_stoch
            h_deter = self.h_rnn(prev_h_stoch_flat, prev_h_deter)
            h_prior_logits = self.h_prior(h_deter)
            h_prior_dist = self.h_dist(h_prior_logits)

            prior_state = MTState(
                deter_h=h_deter,
                deter_l=l_deter,
                distribution_h=h_prior_dist,
                distribution_l=l_prior_dist,
                hidden_h=self.h_rnn.hidden,
                hidden_l=self.l_rnn.hidden,
            )
            priors.append(prior_state)
            prev_state = prior_state

        return stack_mtstates(priors, dim=1)

    def decode_state(self, state: MTState) -> dict[str, Tensor]:  # type: ignore[override]
        """Decode hierarchical state into reconstructions for each modality."""
        vision_recon = self.vision_decoder(state.feature)
        left_tactile_recon = self.left_tactile_decoder(state.feature)
        right_tactile_recon = self.right_tactile_decoder(state.feature)
        return {
            "recon/vision": vision_recon,
            "recon/left_tactile": left_tactile_recon,
            "recon/right_tactile": right_tactile_recon,
        }

    def shared_step(self, batch: tuple[Tensor, ...]) -> dict[str, Tensor]:  # type: ignore[override]
        """Shared training/validation step with hierarchical KL divergence."""
        action_input = batch[0]
        observations = self.get_observations_from_batch(batch)
        initial_observation = self.get_initial_observation(observations)

        posterior, prior = self.rollout_representation(
            actions=action_input,
            observations=observations,
            prev_state=self.initial_state(initial_observation),  # type: ignore[arg-type]
        )

        reconstructions = self.decode_state(posterior)
        targets = self.get_targets_from_batch(batch)

        loss_dict = self.compute_reconstruction_loss(reconstructions, targets)
        kl_div_l = kl_divergence(
            q=posterior.distribution_l.independent(1),
            p=prior.distribution_l.independent(1),
            use_balancing=self.use_kl_balancing,
        ).mul(self.kl_coeff)
        kl_div_h = kl_divergence(
            q=posterior.distribution_h.independent(1),
            p=prior.distribution_h.independent(1),
            use_balancing=self.use_kl_balancing,
        ).mul(self.kl_coeff * self.w_kl_h)

        loss_dict["kl"] = kl_div_l
        loss_dict["kl_h"] = kl_div_h
        loss_dict["loss"] = loss_dict["recon"] + kl_div_l + kl_div_h
        return loss_dict

    encode_observation = MoPoE_MRSSM.encode_observation
