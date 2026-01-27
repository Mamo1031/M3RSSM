"""DataModule for vision-only MTRSSM EpisodeDataset."""

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import StackDataset
from tqdm import tqdm

from models.dataset import (
    BaseEpisodeDataModule,
    BaseEpisodeDataModuleConfig,
    EpisodeDataset,
    Transform,
    load_tensor,
    split_path_list,
)


@dataclass
class EpisodeDataModuleConfig(BaseEpisodeDataModuleConfig):
    """Configuration for vision-only MTRSSM EpisodeDataModule."""

    observation_file_name: str
    observation_preprocess: Transform
    observation_input_transform: Transform
    observation_target_transform: Transform

    def get_observation_file_names(self) -> list[str]:
        """Get the list of observation file names.

        Returns
        -------
        list[str]: The list of observation file names.
        """
        return [self.observation_file_name]

    @staticmethod
    def get_observation_glob_patterns() -> list[str]:
        """Get the list of glob patterns for processed observation files.

        Returns
        -------
        list[str]: The list of glob patterns for processed observation files.
        """
        return ["obs*"]


class EpisodeDataModule(BaseEpisodeDataModule):
    """DataModule for vision-only MTRSSM EpisodeDataset."""

    def __init__(self, config: EpisodeDataModuleConfig) -> None:
        """Initialize the EpisodeDataModule."""
        super().__init__(config)

    def _find_data_paths(self) -> tuple[Path, Path, bool]:
        """Find the paths to observation and action data.

        Returns
        -------
        tuple[Path, Path, bool]: Observation path, action path, and local data flag.
        """
        config = cast("EpisodeDataModuleConfig", self.config)
        data_root = config.data_root
        obs_root = data_root / config.observation_file_name
        act_root = data_root / "joint_states.npy"

        obs_dd = config.data_dir / config.observation_file_name
        act_dd = config.data_dir / "joint_states.npy"

        if obs_root.exists() and act_root.exists():
            return obs_root, act_root, True
        if obs_dd.exists() and act_dd.exists():
            return obs_dd, act_dd, True
        return obs_dd, act_dd, False

    @staticmethod
    def _select_effective_dir(
        config: EpisodeDataModuleConfig,
    ) -> tuple[Path, bool]:
        """Select effective processed data dir and whether vision_obs* is used.

        Returns
        -------
        tuple[Path, bool]: Effective directory and whether to use vision_obs*.
        """
        candidates = [config.processed_data_dir, config.data_root / "processed_data"]
        for candidate in candidates:
            if not candidate.exists():
                continue
            has_actions = bool(list(candidate.glob("act*")))
            has_obs = bool(list(candidate.glob("obs*")))
            has_vision_obs = bool(list(candidate.glob("vision_obs*")))
            if has_actions and (has_obs or has_vision_obs):
                return candidate, has_vision_obs and not has_obs
        return config.processed_data_dir, False

    def _is_processed_data_ready(self) -> bool:
        """Check if processed data already exists.

        Returns
        -------
        bool: True if processed data exists, False otherwise.
        """
        config = cast("EpisodeDataModuleConfig", self.config)
        effective_dir, uses_vision_obs = self._select_effective_dir(config)
        if not effective_dir.exists():
            return False
        has_actions = bool(list(effective_dir.glob("act*")))
        if uses_vision_obs:
            has_observations = bool(list(effective_dir.glob("vision_obs*")))
        else:
            has_observations = bool(list(effective_dir.glob("obs*")))
        return has_actions and has_observations

    def _process_episode_data(self, obs_path: Path, act_path: Path) -> None:  # type: ignore[override]
        """Process episode data from observation and action files."""
        observations = load_tensor(obs_path)  # (N,T,H,W,C) or (N,T,H,W)
        actions = load_tensor(act_path)  # (N,T,A)

        observations = self._normalize_observation_shape(observations)
        num_episodes = observations.shape[0]

        config = cast("EpisodeDataModuleConfig", self.config)
        for i in tqdm(range(num_episodes)):
            action = config.action_preprocess(actions[i])
            observation = config.observation_preprocess(observations[i])  # type: ignore[attr-defined,assignment]
            torch.save(action.detach().clone(), config.processed_data_dir / f"act_{i:03d}.pt")
            torch.save(observation.detach().clone(), config.processed_data_dir / f"obs_{i:03d}.pt")

    def _process_individual_files(self) -> None:
        """Process individual action and observation files."""
        config = cast("EpisodeDataModuleConfig", self.config)
        for action_path in tqdm(sorted(config.data_dir.glob("act*"))):
            action = config.action_preprocess(load_tensor(action_path))
            torch.save(action.detach().clone(), config.processed_data_dir / f"{action_path.stem}.pt")
        for observation_path in tqdm(sorted(config.data_dir.glob("obs*"))):
            observation = config.observation_preprocess(load_tensor(observation_path))
            torch.save(observation.detach().clone(), config.processed_data_dir / f"{observation_path.stem}.pt")

    def setup(self, stage: str = "fit") -> None:
        """Set up the data.

        Raises
        ------
        RuntimeError
            If episode data is missing.
        """
        config = cast("EpisodeDataModuleConfig", self.config)
        effective_dir, uses_vision_obs = self._select_effective_dir(config)
        action_path_list = sorted(effective_dir.glob("act*"))
        if uses_vision_obs:
            observation_path_list = sorted(effective_dir.glob("vision_obs*"))
        else:
            observation_path_list = sorted(effective_dir.glob("obs*"))

        if not action_path_list or not observation_path_list:
            observation_files = ", ".join(config.get_observation_file_names())
            msg = (
                "episode data is missing. Provide raw data and re-run prepare_data().\n"
                f"- Raw data dir: {config.data_dir} (expected: {observation_files}, joint_states.npy)\n"
                f"- Processed data dir: {effective_dir} (expected: act*, obs* or vision_obs*)"
            )
            raise RuntimeError(msg)

        action_path_list, observation_path_list = self._align_episode_paths(
            action_path_list,
            observation_path_list,
        )

        train_action_list, val_action_list = split_path_list(action_path_list, 0.8)
        train_observation_list, val_observation_list = split_path_list(observation_path_list, 0.8)

        if stage == "fit":
            self.train_dataset = StackDataset(
                EpisodeDataset(train_action_list, config.action_input_transform),
                EpisodeDataset(train_observation_list, config.observation_input_transform),
                EpisodeDataset(train_action_list, config.action_target_transform),
                EpisodeDataset(train_observation_list, config.observation_target_transform),
            )
        self.val_dataset = StackDataset(
            EpisodeDataset(val_action_list, config.action_input_transform),
            EpisodeDataset(val_observation_list, config.observation_input_transform),
            EpisodeDataset(val_action_list, config.action_target_transform),
            EpisodeDataset(val_observation_list, config.observation_target_transform),
        )

    @staticmethod
    def _align_episode_paths(
        action_paths: list[Path],
        observation_paths: list[Path],
    ) -> tuple[list[Path], list[Path]]:
        """Align action/observation paths by shared episode id.

        Returns
        -------
        tuple[list[Path], list[Path]]
            Aligned action and observation paths.

        Raises
        ------
        ValueError
            If no matching episode ids are found.
        """
        def extract_id(path: Path) -> str:
            match = re.search(r"(\d+)$", path.stem)
            return match.group(1) if match else path.stem

        action_map = {extract_id(p): p for p in action_paths}
        observation_map = {extract_id(p): p for p in observation_paths}
        common_ids = sorted(
            set(action_map) & set(observation_map),
            key=lambda x: int(x) if x.isdigit() else x,
        )

        if not common_ids:
            msg = "No matching episode ids between action and observation files."
            raise ValueError(msg)

        aligned_actions = [action_map[i] for i in common_ids]
        aligned_observations = [observation_map[i] for i in common_ids]

        if len(aligned_actions) != len(action_paths) or len(aligned_observations) != len(observation_paths):
            warnings.warn(
                f"Warning: Aligning episodes by id. Using {len(common_ids)} pairs "
                f"(actions: {len(action_paths)}, observations: {len(observation_paths)}).",
                stacklevel=2,
            )

        return aligned_actions, aligned_observations
