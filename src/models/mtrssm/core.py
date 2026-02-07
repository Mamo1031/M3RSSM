"""Vision-only MTRSSM module for recurrent state-space model."""

import torch
from distribution_extension import Distribution, MultiOneHotFactory, kl_divergence
from torch import Tensor, nn

from models.core import BaseRSSM
from models.m3rssm.state import MTState, stack_mtstates
from models.networks import Representation, Transition
from models.objective import likelihood


class MTRNN(nn.Module):
    """Multi-timescale RNN cell."""

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


class MTRSSM(BaseRSSM):
    """Vision-only hierarchical MTRSSM (single modality) with action inputs."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        vision_representation: Representation,
        vision_encoder: nn.Module,
        vision_decoder: nn.Module,
        init_proj: nn.Module,
        kl_coeff: float,
        use_kl_balancing: bool,
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
        """Initialize MTRSSM."""
        dummy_transition = Transition(
            deterministic_size=ld_dim,
            hidden_size=ld_dim,
            action_size=1,
            distribution_config=[1, 1],
            activation_name="ELU",
        )
        super().__init__(
            representation=vision_representation,
            transition=dummy_transition,
            init_proj=init_proj,
            kl_coeff=kl_coeff,
            use_kl_balancing=use_kl_balancing,
        )
        self.vision_representation = vision_representation
        self.vision_encoder = vision_encoder
        self.vision_decoder = vision_decoder

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

    def encode_observation(self, observation: Tensor) -> Tensor:  # type: ignore[override]
        """Encode vision observation(s) into embedding."""
        return self.vision_encoder(observation)  # type: ignore[no-any-return]

    def initial_state(self, observation: Tensor) -> MTState:  # type: ignore[override]
        """Initialize hierarchical latent state."""
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
        observations: Tensor,
        prev_state: MTState,
    ) -> tuple[MTState, MTState]:
        """Rollout representation with hierarchical states (vision only)."""
        vision_embed: Tensor = self.vision_encoder(observations)  # type: ignore[no-any-return]
        self._set_prev_hiddens(vision_embed.shape[0], prev_state=prev_state)

        priors: list[MTState] = []
        posteriors: list[MTState] = []

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

            vision_obs_embed = vision_embed[:, t]
            _, vision_logits = self._compute_lower_posterior_with_logits(
                vision_obs_embed,
                l_deter,
                self.vision_representation,
            )
            l_posterior_dist = self.l_dist(vision_logits)
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
            posteriors.append(posterior_state)
            prev_state = posterior_state

        prior = stack_mtstates(priors, dim=1)
        posterior = stack_mtstates(posteriors, dim=1)
        return posterior, prior

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
        """Decode hierarchical state into vision reconstruction."""
        vision_recon = self.vision_decoder(state.feature)
        return {"recon/vision": vision_recon}

    @staticmethod
    def compute_reconstruction_loss(
        reconstructions: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute reconstruction loss for vision modality."""
        vision_recon_loss = likelihood(
            prediction=reconstructions["recon/vision"],
            target=targets["recon/vision"],
            event_ndims=3,
        )
        return {
            "recon": vision_recon_loss,
            "recon/vision": vision_recon_loss,
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

    @staticmethod
    def get_observations_from_batch(batch: tuple[Tensor, ...]) -> Tensor:
        """Extract vision observation sequence from batch."""
        return batch[1]

    @staticmethod
    def get_initial_observation(observations: Tensor) -> Tensor:
        """Extract initial vision observation from sequence."""
        return observations[:, 0]

    @staticmethod
    def get_targets_from_batch(batch: tuple[Tensor, ...]) -> dict[str, Tensor]:
        """Extract reconstruction targets from batch."""
        return {"recon/vision": batch[3]}
