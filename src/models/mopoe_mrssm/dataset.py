"""DataModule for 3-modality MRSSM EpisodeDataset."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py
import torch
from torch import Tensor
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

OBSERVATION_DIM_5D = 5
OBSERVATION_DIM_4D = 4
OBSERVATION_DIM_3D = 3
ACTION_DIM_3D = 3
ACTION_DIM_2D = 2


def identity_transform(data: Tensor) -> Tensor:
    """Return data unchanged."""
    return data


@dataclass
class EpisodeDataModuleConfig(BaseEpisodeDataModuleConfig):
    """Configuration for 3-modality MRSSM EpisodeDataModule."""

    vision_observation_file_name: str
    left_tactile_observation_file_name: str
    right_tactile_observation_file_name: str
    vision_observation_preprocess: Transform
    left_tactile_observation_preprocess: Transform
    right_tactile_observation_preprocess: Transform
    vision_observation_input_transform: Transform
    vision_observation_target_transform: Transform
    left_tactile_observation_input_transform: Transform
    left_tactile_observation_target_transform: Transform
    right_tactile_observation_input_transform: Transform
    right_tactile_observation_target_transform: Transform
    h5_file_name: str | None = None
    h5_vision_key: str = "camera/main"
    h5_left_tactile_key: str = "tactile/left"
    h5_right_tactile_key: str = "tactile/right"
    h5_action_key: str = "arm/leader"

    def get_observation_file_names(self) -> list[str]:
        """Get the list of observation file names.

        Returns
        -------
        list[str]: The list of observation file names.
        """
        return [
            self.vision_observation_file_name,
            self.left_tactile_observation_file_name,
            self.right_tactile_observation_file_name,
        ]

    @staticmethod
    def get_observation_glob_patterns() -> list[str]:
        """Get the list of glob patterns for processed observation files.

        Returns
        -------
        list[str]: The list of glob patterns.
        """
        return [
            "vision_obs*",
            "left_tactile_obs*",
            "right_tactile_obs*",
            "left_tactile_init*",
            "right_tactile_init*",
            "left_tactile_diff_raw*",
            "right_tactile_diff_raw*",
        ]


class EpisodeDataModule(BaseEpisodeDataModule):
    """DataModule for 3-modality MRSSM EpisodeDataset."""

    def __init__(self, config: EpisodeDataModuleConfig) -> None:
        """Initialize the EpisodeDataModule.

        Args:
            config: The configuration for the EpisodeDataModule.
        """
        super().__init__(config)

    def _find_data_paths(self) -> tuple[Path, Path, Path, Path, bool]:
        """Find the paths to vision, left_tactile, right_tactile observations, and action data.

        Returns
        -------
        tuple[Path, Path, Path, Path, bool]
            The vision observation path, left_tactile observation path, right_tactile observation path,
            action path, and whether local data exists.
        """
        config = cast("EpisodeDataModuleConfig", self.config)
        data_root = config.data_root
        vision_obs_root = data_root / config.vision_observation_file_name
        left_tactile_obs_root = data_root / config.left_tactile_observation_file_name
        right_tactile_obs_root = data_root / config.right_tactile_observation_file_name
        act_root = data_root / "joint_states.npy"

        vision_obs_dd = config.data_dir / config.vision_observation_file_name
        left_tactile_obs_dd = config.data_dir / config.left_tactile_observation_file_name
        right_tactile_obs_dd = config.data_dir / config.right_tactile_observation_file_name
        act_dd = config.data_dir / "joint_states.npy"

        if (
            vision_obs_root.exists()
            and left_tactile_obs_root.exists()
            and right_tactile_obs_root.exists()
            and act_root.exists()
        ):
            return vision_obs_root, left_tactile_obs_root, right_tactile_obs_root, act_root, True
        if (
            vision_obs_dd.exists()
            and left_tactile_obs_dd.exists()
            and right_tactile_obs_dd.exists()
            and act_dd.exists()
        ):
            return vision_obs_dd, left_tactile_obs_dd, right_tactile_obs_dd, act_dd, True
        return vision_obs_dd, left_tactile_obs_dd, right_tactile_obs_dd, act_dd, False

    def _process_h5_data(self, h5_path: Path) -> None:
        """Process episode data from an HDF5 file.

        Raises
        ------
        KeyError
            If required datasets are missing from the HDF5 file.
        """
        config = cast("EpisodeDataModuleConfig", self.config)

        with h5py.File(h5_path, "r") as f:
            missing_keys = [
                key
                for key in [
                    config.h5_vision_key,
                    config.h5_left_tactile_key,
                    config.h5_right_tactile_key,
                    config.h5_action_key,
                ]
                if key not in f
            ]
            if missing_keys:
                missing = ", ".join(missing_keys)
                msg = f"HDF5 file is missing required datasets: {missing}"
                raise KeyError(msg)

            vision_observations = torch.tensor(f[config.h5_vision_key][...])
            left_tactile_observations = torch.tensor(f[config.h5_left_tactile_key][...])
            right_tactile_observations = torch.tensor(f[config.h5_right_tactile_key][...])
            actions = torch.tensor(f[config.h5_action_key][...])

        vision_observations = self._ensure_episode_dim(vision_observations, has_channel=True)
        left_tactile_observations = self._ensure_episode_dim(left_tactile_observations, has_channel=True)
        right_tactile_observations = self._ensure_episode_dim(right_tactile_observations, has_channel=True)
        actions = self._ensure_action_episode_dim(actions)

        num_episodes = vision_observations.shape[0]
        for i in tqdm(range(num_episodes)):
            action = config.action_preprocess(actions[i])
            vision_observation = config.vision_observation_preprocess(vision_observations[i])  # type: ignore[attr-defined,assignment]
            left_diff, left_init = self._compute_tactile_diff(left_tactile_observations[i])
            right_diff, right_init = self._compute_tactile_diff(right_tactile_observations[i])
            left_tactile_observation = config.left_tactile_observation_preprocess(left_diff)  # type: ignore[attr-defined,assignment]
            right_tactile_observation = config.right_tactile_observation_preprocess(right_diff)  # type: ignore[attr-defined,assignment]
            torch.save(action.detach().clone(), config.processed_data_dir / f"act_{i:03d}.pt")
            torch.save(
                vision_observation.detach().clone(),
                config.processed_data_dir / f"vision_obs_{i:03d}.pt",
            )
            torch.save(
                left_tactile_observation.detach().clone(),
                config.processed_data_dir / f"left_tactile_obs_{i:03d}.pt",
            )
            torch.save(
                right_tactile_observation.detach().clone(),
                config.processed_data_dir / f"right_tactile_obs_{i:03d}.pt",
            )
            torch.save(
                left_init.detach().clone(),
                config.processed_data_dir / f"left_tactile_init_{i:03d}.pt",
            )
            torch.save(
                right_init.detach().clone(),
                config.processed_data_dir / f"right_tactile_init_{i:03d}.pt",
            )
            torch.save(
                left_diff.detach().clone().float(),
                config.processed_data_dir / f"left_tactile_diff_raw_{i:03d}.pt",
            )
            torch.save(
                right_diff.detach().clone().float(),
                config.processed_data_dir / f"right_tactile_diff_raw_{i:03d}.pt",
            )

    def _processed_data_matches_h5_shape(self, h5_path: Path) -> bool:
        """Check if processed data matches HDF5 spatial shape.

        Returns
        -------
        bool
            True if processed data matches HDF5 spatial shape, False otherwise.
        """
        config = cast("EpisodeDataModuleConfig", self.config)
        effective_dir = config.get_effective_processed_data_dir(config.get_observation_glob_patterns())
        vision_paths = sorted(effective_dir.glob("vision_obs*"))
        if not vision_paths:
            return False
        sample = load_tensor(vision_paths[0])
        if sample.dim() < OBSERVATION_DIM_3D:
            return False
        with h5py.File(h5_path, "r") as f:
            dataset = f[config.h5_vision_key]
            if dataset.ndim < OBSERVATION_DIM_3D:
                return False
            expected_height, expected_width = dataset.shape[-2], dataset.shape[-1]
        return sample.shape[-2] == expected_height and sample.shape[-1] == expected_width

    @staticmethod
    def _clear_processed_files(processed_dir: Path) -> None:
        """Remove processed data files to allow regeneration."""
        for pattern in (
            "act*",
            "vision_obs*",
            "left_tactile_obs*",
            "right_tactile_obs*",
            "left_tactile_init*",
            "right_tactile_init*",
            "left_tactile_diff_raw*",
            "right_tactile_diff_raw*",
        ):
            for path in processed_dir.glob(pattern):
                path.unlink(missing_ok=True)

    @staticmethod
    def _ensure_episode_dim(observations: Tensor, *, has_channel: bool) -> Tensor:
        """Ensure observation tensor has shape (N,T,C,H,W) or (N,T,1,H,W).

        Returns
        -------
        Tensor
            The observation tensor with an episode dimension.

        Raises
        ------
        ValueError
            If the observation tensor shape is unsupported.
        """
        if observations.dim() == OBSERVATION_DIM_5D:
            return observations
        if observations.dim() == OBSERVATION_DIM_4D:
            # (T,C,H,W) -> (1,T,C,H,W)
            return observations.unsqueeze(0)
        if observations.dim() == OBSERVATION_DIM_3D and not has_channel:
            # (T,H,W) -> (1,T,1,H,W)
            return observations.unsqueeze(0).unsqueeze(2)
        msg = f"Unsupported observation shape for HDF5 data: {tuple(observations.shape)}"
        raise ValueError(msg)

    @staticmethod
    def _ensure_action_episode_dim(actions: Tensor) -> Tensor:
        """Ensure action tensor has shape (N,T,A).

        Returns
        -------
        Tensor
            The action tensor with an episode dimension.

        Raises
        ------
        ValueError
            If the action tensor shape is unsupported.
        """
        if actions.dim() == ACTION_DIM_3D:
            return actions
        if actions.dim() == ACTION_DIM_2D:
            return actions.unsqueeze(0)
        msg = f"Unsupported action shape for HDF5 data: {tuple(actions.shape)}"
        raise ValueError(msg)

    def _ensure_processed_h5_data(self) -> None:
        """Ensure processed data is available when HDF5 exists."""
        config = cast("EpisodeDataModuleConfig", self.config)
        h5_name = config.h5_file_name or "rakuda_observations.h5"
        h5_path = config.data_dir / h5_name
        if not h5_path.exists():
            return
        config.processed_data_dir.mkdir(parents=True, exist_ok=True)
        if self._is_processed_data_ready() and self._processed_data_matches_h5_shape(h5_path):
            return
        self._clear_processed_files(config.processed_data_dir)
        self._process_h5_data(h5_path)

    def prepare_data(self) -> None:
        """Prepare the data."""
        config = cast("EpisodeDataModuleConfig", self.config)
        h5_name = config.h5_file_name or "rakuda_observations.h5"
        h5_path = config.data_dir / h5_name
        if h5_path.exists():
            self._ensure_processed_h5_data()
            return

        super().prepare_data()

    def _is_processed_data_ready(self) -> bool:
        """Check if processed data already exists.

        Returns
        -------
        bool
            True if processed data exists, False otherwise.
        """
        config = cast("EpisodeDataModuleConfig", self.config)
        effective_dir = config.get_effective_processed_data_dir(config.get_observation_glob_patterns())
        action_path_list = list(effective_dir.glob("act*"))
        vision_observation_path_list = list(effective_dir.glob("vision_obs*"))
        left_tactile_observation_path_list = list(effective_dir.glob("left_tactile_obs*"))
        right_tactile_observation_path_list = list(effective_dir.glob("right_tactile_obs*"))
        left_tactile_init_path_list = list(effective_dir.glob("left_tactile_init*"))
        right_tactile_init_path_list = list(effective_dir.glob("right_tactile_init*"))
        left_tactile_diff_raw_path_list = list(effective_dir.glob("left_tactile_diff_raw*"))
        right_tactile_diff_raw_path_list = list(effective_dir.glob("right_tactile_diff_raw*"))

        return (
            len(action_path_list) > 0
            and len(vision_observation_path_list) > 0
            and len(left_tactile_observation_path_list) > 0
            and len(right_tactile_observation_path_list) > 0
            and len(left_tactile_init_path_list) > 0
            and len(right_tactile_init_path_list) > 0
            and len(left_tactile_diff_raw_path_list) > 0
            and len(right_tactile_diff_raw_path_list) > 0
        )

    @staticmethod
    def _compute_tactile_diff(observations: Tensor) -> tuple[Tensor, Tensor]:
        """Compute tactile diff and initial frame from raw observations.

        Args:
            observations: Raw tactile observations. Shape: [T,C,H,W] or [T,H,W] or [N,T,C,H,W]

        Returns
        -------
        tuple[Tensor, Tensor]: (diff, initial) where diff is [T,C,H,W] and initial is [C,H,W]
        """
        if observations.dim() == OBSERVATION_DIM_5D:
            observations = observations[0]
        if observations.dim() == OBSERVATION_DIM_3D:
            observations = observations.unsqueeze(1)
        initial = observations[0]
        diff = observations - initial.unsqueeze(0)
        return diff, initial

    def _process_episode_data(
        self,
        vision_obs_path: Path,
        left_tactile_obs_path: Path,
        right_tactile_obs_path: Path,
        act_path: Path,
    ) -> None:  # type: ignore[override]
        """Process episode data from vision, left_tactile, right_tactile observations, and action files.

        Args:
            vision_obs_path: The path to vision observation data.
            left_tactile_obs_path: The path to left tactile observation data.
            right_tactile_obs_path: The path to right tactile observation data.
            act_path: The path to action data.
        """
        vision_observations = load_tensor(vision_obs_path)  # (N,T,C,H,W) or (N,T,H,W)
        left_tactile_observations = load_tensor(left_tactile_obs_path)  # (N,T,C,H,W) or (N,T,H,W)
        right_tactile_observations = load_tensor(right_tactile_obs_path)  # (N,T,C,H,W) or (N,T,H,W)
        actions = load_tensor(act_path)  # (N,T,A)

        vision_observations = self._normalize_observation_shape(vision_observations)
        left_tactile_observations = self._normalize_observation_shape(left_tactile_observations)
        right_tactile_observations = self._normalize_observation_shape(right_tactile_observations)
        num_episodes = vision_observations.shape[0]

        config = cast("EpisodeDataModuleConfig", self.config)
        for i in tqdm(range(num_episodes)):
            action = config.action_preprocess(actions[i])
            vision_observation = config.vision_observation_preprocess(vision_observations[i])  # type: ignore[attr-defined,assignment]
            left_diff, left_init = self._compute_tactile_diff(left_tactile_observations[i])
            right_diff, right_init = self._compute_tactile_diff(right_tactile_observations[i])
            left_tactile_observation = config.left_tactile_observation_preprocess(left_diff)  # type: ignore[attr-defined,assignment]
            right_tactile_observation = config.right_tactile_observation_preprocess(right_diff)  # type: ignore[attr-defined,assignment]
            torch.save(action.detach().clone(), config.processed_data_dir / f"act_{i:03d}.pt")
            torch.save(
                vision_observation.detach().clone(),
                config.processed_data_dir / f"vision_obs_{i:03d}.pt",
            )
            torch.save(
                left_tactile_observation.detach().clone(),
                config.processed_data_dir / f"left_tactile_obs_{i:03d}.pt",
            )
            torch.save(
                right_tactile_observation.detach().clone(),
                config.processed_data_dir / f"right_tactile_obs_{i:03d}.pt",
            )
            torch.save(
                left_init.detach().clone(),
                config.processed_data_dir / f"left_tactile_init_{i:03d}.pt",
            )
            torch.save(
                right_init.detach().clone(),
                config.processed_data_dir / f"right_tactile_init_{i:03d}.pt",
            )

    def _process_individual_files(self) -> None:
        """Process individual action and observation files."""
        config = cast("EpisodeDataModuleConfig", self.config)
        for action_path in tqdm(sorted(config.data_dir.glob("act*"))):
            action = config.action_preprocess(load_tensor(action_path))
            torch.save(action.detach().clone(), config.processed_data_dir / f"{action_path.stem}.pt")
        for vision_observation_path in tqdm(sorted(config.data_dir.glob("vision_obs*"))):
            vision_observation = config.vision_observation_preprocess(load_tensor(vision_observation_path))  # type: ignore[attr-defined,assignment]
            torch.save(
                vision_observation.detach().clone(),
                config.processed_data_dir / f"{vision_observation_path.stem}.pt",
            )
        for left_tactile_observation_path in tqdm(sorted(config.data_dir.glob("left_tactile_obs*"))):
            left_raw = load_tensor(left_tactile_observation_path)
            left_diff, left_init = self._compute_tactile_diff(left_raw)
            left_tactile_observation = config.left_tactile_observation_preprocess(left_diff)  # type: ignore[attr-defined,assignment]
            torch.save(
                left_tactile_observation.detach().clone(),
                config.processed_data_dir / f"{left_tactile_observation_path.stem}.pt",
            )
            left_init_name = left_tactile_observation_path.stem.replace("left_tactile_obs", "left_tactile_init")
            torch.save(left_init.detach().clone(), config.processed_data_dir / f"{left_init_name}.pt")
            left_diff_name = left_tactile_observation_path.stem.replace(
                "left_tactile_obs", "left_tactile_diff_raw",
            )
            torch.save(left_diff.detach().clone().float(), config.processed_data_dir / f"{left_diff_name}.pt")
        for right_tactile_observation_path in tqdm(sorted(config.data_dir.glob("right_tactile_obs*"))):
            right_raw = load_tensor(right_tactile_observation_path)
            right_diff, right_init = self._compute_tactile_diff(right_raw)
            right_tactile_observation = config.right_tactile_observation_preprocess(right_diff)  # type: ignore[attr-defined,assignment]
            torch.save(
                right_tactile_observation.detach().clone(),
                config.processed_data_dir / f"{right_tactile_observation_path.stem}.pt",
            )
            right_init_name = right_tactile_observation_path.stem.replace("right_tactile_obs", "right_tactile_init")
            torch.save(right_init.detach().clone(), config.processed_data_dir / f"{right_init_name}.pt")
            right_diff_name = right_tactile_observation_path.stem.replace(
                "right_tactile_obs", "right_tactile_diff_raw",
            )
            torch.save(right_diff.detach().clone().float(), config.processed_data_dir / f"{right_diff_name}.pt")

    def setup(self, stage: str = "fit") -> None:
        """Set up the data."""
        config = cast("EpisodeDataModuleConfig", self.config)
        self._ensure_processed_h5_data()
        effective_dir = config.get_effective_processed_data_dir(config.get_observation_glob_patterns())
        action_path_list = sorted(effective_dir.glob("act*"))
        vision_observation_path_list = sorted(effective_dir.glob("vision_obs*"))
        left_tactile_observation_path_list = sorted(effective_dir.glob("left_tactile_obs*"))
        right_tactile_observation_path_list = sorted(effective_dir.glob("right_tactile_obs*"))
        left_tactile_init_path_list = sorted(effective_dir.glob("left_tactile_init*"))
        right_tactile_init_path_list = sorted(effective_dir.glob("right_tactile_init*"))
        left_tactile_diff_raw_path_list = sorted(effective_dir.glob("left_tactile_diff_raw*"))
        right_tactile_diff_raw_path_list = sorted(effective_dir.glob("right_tactile_diff_raw*"))

        train_action_list, val_action_list = split_path_list(action_path_list, 0.8)
        train_vision_observation_list, val_vision_observation_list = split_path_list(vision_observation_path_list, 0.8)
        train_left_tactile_observation_list, val_left_tactile_observation_list = split_path_list(
            left_tactile_observation_path_list,
            0.8,
        )
        train_right_tactile_observation_list, val_right_tactile_observation_list = split_path_list(
            right_tactile_observation_path_list,
            0.8,
        )
        train_left_tactile_init_list, val_left_tactile_init_list = split_path_list(
            left_tactile_init_path_list,
            0.8,
        )
        train_right_tactile_init_list, val_right_tactile_init_list = split_path_list(
            right_tactile_init_path_list,
            0.8,
        )
        train_left_tactile_diff_raw_list, val_left_tactile_diff_raw_list = split_path_list(
            left_tactile_diff_raw_path_list,
            0.8,
        )
        train_right_tactile_diff_raw_list, val_right_tactile_diff_raw_list = split_path_list(
            right_tactile_diff_raw_path_list,
            0.8,
        )

        if stage == "fit":
            self.train_dataset = StackDataset(
                EpisodeDataset(train_action_list, config.action_input_transform),
                EpisodeDataset(train_vision_observation_list, config.vision_observation_input_transform),
                EpisodeDataset(train_left_tactile_observation_list, config.left_tactile_observation_input_transform),
                EpisodeDataset(train_right_tactile_observation_list, config.right_tactile_observation_input_transform),
                EpisodeDataset(train_action_list, config.action_target_transform),
                EpisodeDataset(train_vision_observation_list, config.vision_observation_target_transform),
                EpisodeDataset(train_left_tactile_observation_list, config.left_tactile_observation_target_transform),
                EpisodeDataset(train_right_tactile_observation_list, config.right_tactile_observation_target_transform),
                EpisodeDataset(train_left_tactile_init_list, identity_transform),
                EpisodeDataset(train_right_tactile_init_list, identity_transform),
                EpisodeDataset(train_left_tactile_diff_raw_list, identity_transform),
                EpisodeDataset(train_right_tactile_diff_raw_list, identity_transform),
            )
        self.val_dataset = StackDataset(
            EpisodeDataset(val_action_list, config.action_input_transform),
            EpisodeDataset(val_vision_observation_list, config.vision_observation_input_transform),
            EpisodeDataset(val_left_tactile_observation_list, config.left_tactile_observation_input_transform),
            EpisodeDataset(val_right_tactile_observation_list, config.right_tactile_observation_input_transform),
            EpisodeDataset(val_action_list, config.action_target_transform),
            EpisodeDataset(val_vision_observation_list, config.vision_observation_target_transform),
            EpisodeDataset(val_left_tactile_observation_list, config.left_tactile_observation_target_transform),
            EpisodeDataset(val_right_tactile_observation_list, config.right_tactile_observation_target_transform),
            EpisodeDataset(val_left_tactile_init_list, identity_transform),
            EpisodeDataset(val_right_tactile_init_list, identity_transform),
            EpisodeDataset(val_left_tactile_diff_raw_list, identity_transform),
            EpisodeDataset(val_right_tactile_diff_raw_list, identity_transform),
        )
