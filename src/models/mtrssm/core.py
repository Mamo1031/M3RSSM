"""Vision-only MTRSSM module for recurrent state-space model."""

from torch import Tensor, nn

from models.core import BaseRSSM
from models.networks import Representation, Transition
from models.objective import likelihood
from models.state import State


class MTRSSM(BaseRSSM):  # noqa: N801
    """Vision-only MTRSSM (single modality) with action inputs."""

    def __init__(
        self,
        *,
        vision_representation: Representation,
        transition: Transition,
        vision_encoder: nn.Module,
        vision_decoder: nn.Module,
        init_proj: nn.Module,
        kl_coeff: float,
        use_kl_balancing: bool,
    ) -> None:
        """Initialize MTRSSM.

        Args:
            vision_representation: Representation network for vision modality.
            transition: Transition network (RSSM prior).
            vision_encoder: Vision encoder network.
            vision_decoder: Vision decoder network.
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
        self.vision_encoder = vision_encoder
        self.vision_decoder = vision_decoder

    def encode_observation(self, observation: Tensor) -> Tensor:  # type: ignore[override]
        """Encode vision observation(s) into embedding."""
        return self.vision_encoder(observation)  # type: ignore[no-any-return]

    def decode_state(self, state: State) -> dict[str, Tensor]:
        """Decode state into vision reconstruction."""
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
