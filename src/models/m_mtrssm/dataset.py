"""DataModule for M_MTRSSM: re-exports 3-modality dataset from mopoe_mrssm."""

from models.mopoe_mrssm.dataset import EpisodeDataModule, EpisodeDataModuleConfig

__all__ = ["EpisodeDataModule", "EpisodeDataModuleConfig"]
