"""State utilities for M3RSSM (hierarchical states)."""

from collections.abc import Generator

import torch
from distribution_extension import Distribution
from distribution_extension.utils import cat_distribution, stack_distribution
from torch import Tensor


class MTState:
    """Hierarchical latent state with higher and lower layers."""

    def __init__(
        self,
        *,
        deter_h: Tensor,
        deter_l: Tensor,
        distribution_h: Distribution,
        distribution_l: Distribution,
        hidden_h: Tensor,
        hidden_l: Tensor,
        stoch_h: Tensor | None = None,
        stoch_l: Tensor | None = None,
    ) -> None:
        """Initialize MTState."""
        self.deter_h = deter_h
        self.deter_l = deter_l
        self.distribution_h = distribution_h
        self.distribution_l = distribution_l
        self.hidden_h = hidden_h
        self.hidden_l = hidden_l
        self.stoch_h = distribution_h.rsample() if stoch_h is None else stoch_h
        self.stoch_l = distribution_l.rsample() if stoch_l is None else stoch_l
        self.feature = torch.cat([self.deter_h, self.stoch_h, self.deter_l, self.stoch_l], dim=-1)

    def __iter__(self) -> Generator["MTState", None, None]:
        """Iterate over batch dimension."""
        for i in range(self.deter_h.shape[0]):
            yield self[i]

    def __getitem__(self, loc: slice | int | tuple[slice | int, ...]) -> "MTState":
        """Index the state."""
        return type(self)(
            deter_h=self.deter_h[loc],
            deter_l=self.deter_l[loc],
            distribution_h=self.distribution_h[loc],
            distribution_l=self.distribution_l[loc],
            hidden_h=self.hidden_h[loc] if self.hidden_h.dim() > 1 else self.hidden_h,
            hidden_l=self.hidden_l[loc] if self.hidden_l.dim() > 1 else self.hidden_l,
            stoch_h=self.stoch_h[loc],
            stoch_l=self.stoch_l[loc],
        )

    def to(self, device: torch.device) -> "MTState":
        """Move the state to the device."""
        return type(self)(
            deter_h=self.deter_h.to(device),
            deter_l=self.deter_l.to(device),
            distribution_h=self.distribution_h.to(device),
            distribution_l=self.distribution_l.to(device),
            hidden_h=self.hidden_h.to(device),
            hidden_l=self.hidden_l.to(device),
            stoch_h=self.stoch_h.to(device),
            stoch_l=self.stoch_l.to(device),
        )

    def detach(self) -> "MTState":
        """Detach the state."""
        return type(self)(
            deter_h=self.deter_h.detach(),
            deter_l=self.deter_l.detach(),
            distribution_h=self.distribution_h.detach(),
            distribution_l=self.distribution_l.detach(),
            hidden_h=self.hidden_h.detach(),
            hidden_l=self.hidden_l.detach(),
            stoch_h=self.stoch_h.detach(),
            stoch_l=self.stoch_l.detach(),
        )

    def clone(self) -> "MTState":
        """Clone the state."""
        return type(self)(
            deter_h=self.deter_h.clone(),
            deter_l=self.deter_l.clone(),
            distribution_h=self.distribution_h.clone(),
            distribution_l=self.distribution_l.clone(),
            hidden_h=self.hidden_h.clone(),
            hidden_l=self.hidden_l.clone(),
            stoch_h=self.stoch_h.clone(),
            stoch_l=self.stoch_l.clone(),
        )

    def squeeze(self, dim: int) -> "MTState":
        """Squeeze the state."""
        return type(self)(
            deter_h=self.deter_h.squeeze(dim),
            deter_l=self.deter_l.squeeze(dim),
            distribution_h=self.distribution_h.squeeze(dim),
            distribution_l=self.distribution_l.squeeze(dim),
            hidden_h=self.hidden_h.squeeze(dim) if self.hidden_h.dim() > 1 else self.hidden_h,
            hidden_l=self.hidden_l.squeeze(dim) if self.hidden_l.dim() > 1 else self.hidden_l,
            stoch_h=self.stoch_h.squeeze(dim),
            stoch_l=self.stoch_l.squeeze(dim),
        )

    def unsqueeze(self, dim: int) -> "MTState":
        """Unsqueeze the state."""
        return type(self)(
            deter_h=self.deter_h.unsqueeze(dim),
            deter_l=self.deter_l.unsqueeze(dim),
            distribution_h=self.distribution_h.unsqueeze(dim),
            distribution_l=self.distribution_l.unsqueeze(dim),
            hidden_h=self.hidden_h.unsqueeze(dim) if self.hidden_h.dim() > 1 else self.hidden_h,
            hidden_l=self.hidden_l.unsqueeze(dim) if self.hidden_l.dim() > 1 else self.hidden_l,
            stoch_h=self.stoch_h.unsqueeze(dim),
            stoch_l=self.stoch_l.unsqueeze(dim),
        )


def stack_mtstates(states: list[MTState], dim: int) -> MTState:
    """Stack MTStates along a dimension."""
    deter_h = torch.stack([state.deter_h for state in states], dim=dim)
    deter_l = torch.stack([state.deter_l for state in states], dim=dim)
    stoch_h = torch.stack([state.stoch_h for state in states], dim=dim)
    stoch_l = torch.stack([state.stoch_l for state in states], dim=dim)
    distribution_h = stack_distribution([state.distribution_h for state in states], dim)
    distribution_l = stack_distribution([state.distribution_l for state in states], dim)
    hidden_h = (
        torch.stack([state.hidden_h for state in states], dim=dim) if states[0].hidden_h.dim() > 1 else states[0].hidden_h
    )
    hidden_l = (
        torch.stack([state.hidden_l for state in states], dim=dim) if states[0].hidden_l.dim() > 1 else states[0].hidden_l
    )
    return MTState(
        deter_h=deter_h,
        deter_l=deter_l,
        distribution_h=distribution_h,
        distribution_l=distribution_l,
        hidden_h=hidden_h,
        hidden_l=hidden_l,
        stoch_h=stoch_h,
        stoch_l=stoch_l,
    )


def cat_mtstates(states: list[MTState], dim: int) -> MTState:
    """Concatenate MTStates along a dimension."""
    deter_h = torch.cat([state.deter_h for state in states], dim=dim)
    deter_l = torch.cat([state.deter_l for state in states], dim=dim)
    stoch_h = torch.cat([state.stoch_h for state in states], dim=dim)
    stoch_l = torch.cat([state.stoch_l for state in states], dim=dim)
    distribution_h = cat_distribution([state.distribution_h for state in states], dim)
    distribution_l = cat_distribution([state.distribution_l for state in states], dim)
    hidden_h = states[-1].hidden_h
    hidden_l = states[-1].hidden_l
    return MTState(
        deter_h=deter_h,
        deter_l=deter_l,
        distribution_h=distribution_h,
        distribution_l=distribution_l,
        hidden_h=hidden_h,
        hidden_l=hidden_l,
        stoch_h=stoch_h,
        stoch_l=stoch_l,
    )
