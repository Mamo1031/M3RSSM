"""Tests of `mopoe_mrssm/dataset.py`."""

from pathlib import Path

import h5py
import pytest
import torch

from models.mopoe_mrssm.dataset import EpisodeDataModule, EpisodeDataModuleConfig


def identity_transform(x: torch.Tensor) -> torch.Tensor:
    """Identity transform."""
    return x


def test__ensure_episode_dim_5d() -> None:
    """Test _ensure_episode_dim with 5D input."""
    obs = torch.rand(2, 5, 3, 64, 64)
    result = EpisodeDataModule._ensure_episode_dim(obs, has_channel=True)
    assert result.shape == (2, 5, 3, 64, 64)


def test__ensure_episode_dim_4d() -> None:
    """Test _ensure_episode_dim with 4D input."""
    obs = torch.rand(5, 3, 64, 64)
    result = EpisodeDataModule._ensure_episode_dim(obs, has_channel=True)
    assert result.shape == (1, 5, 3, 64, 64)


def test__ensure_episode_dim_3d_no_channel() -> None:
    """Test _ensure_episode_dim with 3D input (no channel)."""
    obs = torch.rand(5, 64, 64)
    result = EpisodeDataModule._ensure_episode_dim(obs, has_channel=False)
    assert result.shape == (1, 5, 1, 64, 64)


def test__ensure_episode_dim_unsupported() -> None:
    """Test _ensure_episode_dim with unsupported shape."""
    obs = torch.rand(3, 4)
    with pytest.raises(ValueError, match="Unsupported observation shape"):
        EpisodeDataModule._ensure_episode_dim(obs, has_channel=True)


def test__ensure_action_episode_dim_3d() -> None:
    """Test _ensure_action_episode_dim with 3D input."""
    actions = torch.rand(2, 5, 15)
    result = EpisodeDataModule._ensure_action_episode_dim(actions)
    assert result.shape == (2, 5, 15)


def test__ensure_action_episode_dim_2d() -> None:
    """Test _ensure_action_episode_dim with 2D input."""
    actions = torch.rand(5, 15)
    result = EpisodeDataModule._ensure_action_episode_dim(actions)
    assert result.shape == (1, 5, 15)


def test__ensure_action_episode_dim_unsupported() -> None:
    """Test _ensure_action_episode_dim with unsupported shape."""
    actions = torch.rand(3)
    with pytest.raises(ValueError, match="Unsupported action shape"):
        EpisodeDataModule._ensure_action_episode_dim(actions)


def test__compute_tactile_diff_5d() -> None:
    """Test _compute_tactile_diff with 5D input."""
    obs = torch.rand(1, 5, 1, 8, 8)
    diff, init = EpisodeDataModule._compute_tactile_diff(obs)
    assert diff.shape == (5, 1, 8, 8)
    assert init.shape == (1, 8, 8)


def test__compute_tactile_diff_4d() -> None:
    """Test _compute_tactile_diff with 4D input (T,C,H,W)."""
    obs = torch.rand(5, 1, 8, 8)
    diff, init = EpisodeDataModule._compute_tactile_diff(obs)
    assert diff.shape == (5, 1, 8, 8)
    assert init.shape == (1, 8, 8)


def test__compute_tactile_diff_3d() -> None:
    """Test _compute_tactile_diff with 3D input (T,H,W)."""
    obs = torch.rand(5, 8, 8)
    diff, init = EpisodeDataModule._compute_tactile_diff(obs)
    assert diff.shape == (5, 1, 8, 8)
    assert init.shape == (1, 8, 8)


def test__clear_processed_files(tmp_path: Path) -> None:
    """Test _clear_processed_files removes processed files."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    (processed_dir / "act_000.pt").touch()
    (processed_dir / "vision_obs_000.pt").touch()

    EpisodeDataModule._clear_processed_files(processed_dir)
    assert not list(processed_dir.glob("act*"))
    assert not list(processed_dir.glob("vision_obs*"))


def test__process_h5_data_key_error(tmp_path: Path) -> None:
    """Test _process_h5_data raises KeyError when datasets are missing."""
    h5_path = tmp_path / "test.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("camera/main", data=torch.rand(1, 5, 3, 64, 64).numpy())

    class TestConfig(EpisodeDataModuleConfig):
        @property
        def data_root(self) -> Path:
            return tmp_path

        @property
        def data_dir(self) -> Path:
            return tmp_path

        @property
        def processed_data_dir(self) -> Path:
            return tmp_path / "processed"

    config = TestConfig(
        data_name="x",
        batch_size=2,
        num_workers=0,
        gdrive_url="https://example.com",
        vision_observation_file_name="v",
        left_tactile_observation_file_name="l",
        right_tactile_observation_file_name="r",
        vision_observation_preprocess=identity_transform,
        left_tactile_observation_preprocess=identity_transform,
        right_tactile_observation_preprocess=identity_transform,
        vision_observation_input_transform=identity_transform,
        vision_observation_target_transform=identity_transform,
        left_tactile_observation_input_transform=identity_transform,
        left_tactile_observation_target_transform=identity_transform,
        right_tactile_observation_input_transform=identity_transform,
        right_tactile_observation_target_transform=identity_transform,
        action_preprocess=identity_transform,
        action_input_transform=identity_transform,
        action_target_transform=identity_transform,
        h5_file_name="test.h5",
    )
    dm = EpisodeDataModule(config)
    with pytest.raises(KeyError, match="tactile/left"):
        dm._process_h5_data(h5_path)


def test__process_h5_data_success(tmp_path: Path) -> None:
    """Test _process_h5_data processes valid HDF5 file."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    h5_path = data_dir / "rakuda.h5"

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("camera/main", data=torch.rand(2, 5, 3, 64, 64).numpy())
        f.create_dataset("tactile/left", data=torch.rand(2, 5, 1, 8, 8).numpy())
        f.create_dataset("tactile/right", data=torch.rand(2, 5, 1, 8, 8).numpy())
        f.create_dataset("arm/leader", data=torch.rand(2, 5, 15).numpy())

    class TestConfig(EpisodeDataModuleConfig):
        @property
        def data_root(self) -> Path:
            return tmp_path

        @property
        def data_dir(self) -> Path:
            return data_dir

        @property
        def processed_data_dir(self) -> Path:
            return processed_dir

    config = TestConfig(
        data_name="x",
        batch_size=2,
        num_workers=0,
        gdrive_url="https://example.com",
        vision_observation_file_name="v",
        left_tactile_observation_file_name="l",
        right_tactile_observation_file_name="r",
        vision_observation_preprocess=identity_transform,
        left_tactile_observation_preprocess=identity_transform,
        right_tactile_observation_preprocess=identity_transform,
        vision_observation_input_transform=identity_transform,
        vision_observation_target_transform=identity_transform,
        left_tactile_observation_input_transform=identity_transform,
        left_tactile_observation_target_transform=identity_transform,
        right_tactile_observation_input_transform=identity_transform,
        right_tactile_observation_target_transform=identity_transform,
        action_preprocess=identity_transform,
        action_input_transform=identity_transform,
        action_target_transform=identity_transform,
        h5_file_name="rakuda.h5",
    )
    dm = EpisodeDataModule(config)
    dm._process_h5_data(h5_path)
    assert len(list(processed_dir.glob("act_*"))) == 2
    assert len(list(processed_dir.glob("vision_obs_*"))) == 2


def test__processed_data_matches_h5_shape(tmp_path: Path) -> None:
    """Test _processed_data_matches_h5_shape."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    torch.save(torch.rand(2, 5, 3, 64, 64), processed_dir / "vision_obs_000.pt")

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    h5_path = data_dir / "test.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("camera/main", data=torch.rand(1, 5, 3, 64, 64).numpy())

    class TestConfig(EpisodeDataModuleConfig):
        @property
        def data_root(self) -> Path:
            return tmp_path

        @property
        def processed_data_dir(self) -> Path:
            return processed_dir

    config = TestConfig(
        data_name="x",
        batch_size=2,
        num_workers=0,
        gdrive_url="https://example.com",
        vision_observation_file_name="v",
        left_tactile_observation_file_name="l",
        right_tactile_observation_file_name="r",
        vision_observation_preprocess=identity_transform,
        left_tactile_observation_preprocess=identity_transform,
        right_tactile_observation_preprocess=identity_transform,
        vision_observation_input_transform=identity_transform,
        vision_observation_target_transform=identity_transform,
        left_tactile_observation_input_transform=identity_transform,
        left_tactile_observation_target_transform=identity_transform,
        right_tactile_observation_input_transform=identity_transform,
        right_tactile_observation_target_transform=identity_transform,
        action_preprocess=identity_transform,
        action_input_transform=identity_transform,
        action_target_transform=identity_transform,
    )
    dm = EpisodeDataModule(config)
    assert dm._processed_data_matches_h5_shape(h5_path) is True


def test__mopoe_setup_and_dataloaders(tmp_path: Path) -> None:
    """Test MoPoE EpisodeDataModule setup and dataloaders with processed data."""
    processed_dir = tmp_path / "processed_mopoe"
    processed_dir.mkdir()
    for i in range(5):
        torch.save(torch.rand(5, 15), processed_dir / f"act_{i:03d}.pt")
        torch.save(torch.rand(5, 3, 64, 64), processed_dir / f"vision_obs_{i:03d}.pt")
        torch.save(torch.rand(5, 1, 8, 8), processed_dir / f"left_tactile_obs_{i:03d}.pt")
        torch.save(torch.rand(5, 1, 8, 8), processed_dir / f"right_tactile_obs_{i:03d}.pt")
        torch.save(torch.rand(1, 8, 8), processed_dir / f"left_tactile_init_{i:03d}.pt")
        torch.save(torch.rand(1, 8, 8), processed_dir / f"right_tactile_init_{i:03d}.pt")
        torch.save(torch.rand(5, 1, 8, 8).float(), processed_dir / f"left_tactile_diff_raw_{i:03d}.pt")
        torch.save(torch.rand(5, 1, 8, 8).float(), processed_dir / f"right_tactile_diff_raw_{i:03d}.pt")

    class TestConfig(EpisodeDataModuleConfig):
        @property
        def data_root(self) -> Path:
            return tmp_path

        @property
        def data_dir(self) -> Path:
            return tmp_path / "data"

        @property
        def processed_data_dir(self) -> Path:
            return processed_dir

    config = TestConfig(
        data_name="mopoe",
        batch_size=2,
        num_workers=1,
        gdrive_url="https://example.com",
        vision_observation_file_name="v",
        left_tactile_observation_file_name="l",
        right_tactile_observation_file_name="r",
        vision_observation_preprocess=identity_transform,
        left_tactile_observation_preprocess=identity_transform,
        right_tactile_observation_preprocess=identity_transform,
        vision_observation_input_transform=identity_transform,
        vision_observation_target_transform=identity_transform,
        left_tactile_observation_input_transform=identity_transform,
        left_tactile_observation_target_transform=identity_transform,
        right_tactile_observation_input_transform=identity_transform,
        right_tactile_observation_target_transform=identity_transform,
        action_preprocess=identity_transform,
        action_input_transform=identity_transform,
        action_target_transform=identity_transform,
    )
    dm = EpisodeDataModule(config)
    dm._ensure_processed_h5_data()
    dm.setup("fit")
    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    batch = next(iter(dm.train_dataloader()))
    assert len(batch) == 12
