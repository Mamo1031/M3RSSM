"""Tests of `mtrssm/dataset.py`."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from models.mtrssm.dataset import EpisodeDataModule, EpisodeDataModuleConfig


def identity_transform(x: torch.Tensor) -> torch.Tensor:
    """Identity transform."""
    return x


@pytest.fixture
def mtrssm_dataset_config(tmp_path: Path) -> EpisodeDataModuleConfig:
    """Create MTRSSM dataset config with tmp data dir."""
    processed_dir = tmp_path / "processed_test_data"
    processed_dir.mkdir()
    # Create minimal episode data (need 5+ for 80/20 split to give train items)
    for i in range(5):
        torch.save(torch.rand(5, 15), processed_dir / f"act_{i:03d}.pt")
        torch.save(torch.rand(5, 3, 64, 64), processed_dir / f"obs_{i:03d}.pt")

    class TestConfig(EpisodeDataModuleConfig):
        @property
        def data_root(self) -> Path:
            return tmp_path

        @property
        def processed_data_dir(self) -> Path:
            return tmp_path / "processed_test_data"

        @property
        def data_dir(self) -> Path:
            return tmp_path / "test_data"

    return TestConfig(
        data_name="test_data",
        batch_size=2,
        num_workers=1,
        gdrive_url="https://example.com",
        observation_file_name="obs.npy",
        observation_preprocess=identity_transform,
        observation_input_transform=identity_transform,
        observation_target_transform=identity_transform,
        action_preprocess=identity_transform,
        action_input_transform=identity_transform,
        action_target_transform=identity_transform,
    )


def test__mtrssm_align_episode_paths(tmp_path: Path) -> None:
    """Test `_align_episode_paths` static method."""
    base = tmp_path / "episodes"
    base.mkdir()
    action_paths = [base / "act_001.pt", base / "act_002.pt"]
    observation_paths = [base / "obs_001.pt", base / "obs_002.pt"]
    aligned_actions, aligned_observations = EpisodeDataModule._align_episode_paths(
        action_paths,
        observation_paths,
    )
    assert len(aligned_actions) == 2
    assert len(aligned_observations) == 2
    assert aligned_actions[0].stem == "act_001"


def test__mtrssm_align_episode_paths_mismatch(tmp_path: Path) -> None:
    """Test `_align_episode_paths` with no matching ids."""
    base = tmp_path / "episodes"
    base.mkdir()
    action_paths = [base / "act_001.pt"]
    observation_paths = [base / "obs_999.pt"]
    with pytest.raises(ValueError, match=r"No matching episode ids"):
        EpisodeDataModule._align_episode_paths(action_paths, observation_paths)


def test__mtrssm_select_effective_dir(tmp_path: Path) -> None:
    """Test `_select_effective_dir` with obs files."""
    processed_dir = tmp_path / "processed_test"
    processed_dir.mkdir()
    (processed_dir / "act_000.pt").touch()
    (processed_dir / "obs_000.pt").touch()

    class TestConfig(EpisodeDataModuleConfig):
        @property
        def data_root(self) -> Path:
            return tmp_path

        @property
        def processed_data_dir(self) -> Path:
            return processed_dir

    config = TestConfig(
        data_name="test",
        batch_size=2,
        num_workers=1,
        gdrive_url="https://example.com",
        observation_file_name="obs",
        observation_preprocess=identity_transform,
        observation_input_transform=identity_transform,
        observation_target_transform=identity_transform,
        action_preprocess=identity_transform,
        action_input_transform=identity_transform,
        action_target_transform=identity_transform,
    )
    effective_dir, uses_vision = EpisodeDataModule._select_effective_dir(
        cast("EpisodeDataModuleConfig", config),
    )
    assert effective_dir == processed_dir
    assert uses_vision is False


def test__mtrssm_select_effective_dir_vision_obs(tmp_path: Path) -> None:
    """Test `_select_effective_dir` with vision_obs files (no obs)."""
    processed_dir = tmp_path / "processed_vision"
    processed_dir.mkdir()
    (processed_dir / "act_000.pt").touch()
    (processed_dir / "vision_obs_000.pt").touch()

    class TestConfig(EpisodeDataModuleConfig):
        @property
        def data_root(self) -> Path:
            return tmp_path

        @property
        def processed_data_dir(self) -> Path:
            return processed_dir

    config = TestConfig(
        data_name="test",
        batch_size=2,
        num_workers=1,
        gdrive_url="https://example.com",
        observation_file_name="obs",
        observation_preprocess=identity_transform,
        observation_input_transform=identity_transform,
        observation_target_transform=identity_transform,
        action_preprocess=identity_transform,
        action_input_transform=identity_transform,
        action_target_transform=identity_transform,
    )
    effective_dir, uses_vision = EpisodeDataModule._select_effective_dir(
        cast("EpisodeDataModuleConfig", config),
    )
    assert effective_dir == processed_dir
    assert uses_vision is True


def test__mtrssm_is_processed_data_ready(tmp_path: Path) -> None:
    """Test `_is_processed_data_ready` when data exists."""
    processed_dir = tmp_path / "processed_test"
    processed_dir.mkdir()
    (processed_dir / "act_000.pt").touch()
    (processed_dir / "obs_000.pt").touch()

    class TestConfig(EpisodeDataModuleConfig):
        @property
        def data_root(self) -> Path:
            return tmp_path

        @property
        def processed_data_dir(self) -> Path:
            return processed_dir

    config = TestConfig(
        data_name="test",
        batch_size=2,
        num_workers=1,
        gdrive_url="https://example.com",
        observation_file_name="obs",
        observation_preprocess=identity_transform,
        observation_input_transform=identity_transform,
        observation_target_transform=identity_transform,
        action_preprocess=identity_transform,
        action_input_transform=identity_transform,
        action_target_transform=identity_transform,
    )
    dm = EpisodeDataModule(config)
    assert dm._is_processed_data_ready() is True


def test__mtrssm_setup(mtrssm_dataset_config: EpisodeDataModuleConfig) -> None:
    """Test MTRSSM EpisodeDataModule setup."""
    dm = EpisodeDataModule(mtrssm_dataset_config)
    dm.setup("fit")
    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert len(dm.train_dataset) > 0  # type: ignore[arg-type]
    assert len(dm.val_dataset) > 0  # type: ignore[arg-type]


def test__mtrssm_train_dataloader(mtrssm_dataset_config: EpisodeDataModuleConfig) -> None:
    """Test MTRSSM train dataloader."""
    dm = EpisodeDataModule(mtrssm_dataset_config)
    dm.setup("fit")
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert len(batch) == 4
    assert batch[0].shape[0] <= 2
    assert batch[1].shape[0] <= 2


def test__mtrssm_val_dataloader(mtrssm_dataset_config: EpisodeDataModuleConfig) -> None:
    """Test MTRSSM val dataloader."""
    dm = EpisodeDataModule(mtrssm_dataset_config)
    dm.setup("fit")
    loader = dm.val_dataloader()
    batch = next(iter(loader))
    assert len(batch) == 4


def test__mtrssm_find_data_paths_data_root(tmp_path: Path) -> None:
    """Test _find_data_paths returns data_root paths when they exist."""
    rng = np.random.default_rng()
    obs_file = tmp_path / "obs.npy"
    act_file = tmp_path / "joint_states.npy"
    np.save(obs_file, rng.random((2, 5, 64, 64)).astype(np.float32))
    np.save(act_file, rng.random((2, 5, 15)).astype(np.float32))

    class TestConfig(EpisodeDataModuleConfig):
        @property
        def data_root(self) -> Path:
            return tmp_path

        @property
        def data_dir(self) -> Path:
            return tmp_path / "nonexistent"

        @property
        def processed_data_dir(self) -> Path:
            return tmp_path / "processed"

    config = TestConfig(
        data_name="x",
        batch_size=2,
        num_workers=1,
        gdrive_url="https://example.com",
        observation_file_name="obs.npy",
        observation_preprocess=identity_transform,
        observation_input_transform=identity_transform,
        observation_target_transform=identity_transform,
        action_preprocess=identity_transform,
        action_input_transform=identity_transform,
        action_target_transform=identity_transform,
    )
    dm = EpisodeDataModule(config)
    obs_p, act_p, has_local = dm._find_data_paths()
    assert has_local is True
    assert obs_p == obs_file
    assert act_p == act_file


def test__mtrssm_find_data_paths_data_dir(tmp_path: Path) -> None:
    """Test _find_data_paths returns data_dir paths when data_root has no files."""
    rng = np.random.default_rng()
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    obs_file = data_dir / "obs.npy"
    act_file = data_dir / "joint_states.npy"
    np.save(obs_file, rng.random((2, 5, 64, 64)).astype(np.float32))
    np.save(act_file, rng.random((2, 5, 15)).astype(np.float32))

    class TestConfig(EpisodeDataModuleConfig):
        @property
        def data_root(self) -> Path:
            return tmp_path

        @property
        def data_dir(self) -> Path:
            return data_dir

        @property
        def processed_data_dir(self) -> Path:
            return tmp_path / "processed"

    config = TestConfig(
        data_name="test_data",
        batch_size=2,
        num_workers=1,
        gdrive_url="https://example.com",
        observation_file_name="obs.npy",
        observation_preprocess=identity_transform,
        observation_input_transform=identity_transform,
        observation_target_transform=identity_transform,
        action_preprocess=identity_transform,
        action_input_transform=identity_transform,
        action_target_transform=identity_transform,
    )
    dm = EpisodeDataModule(config)
    obs_p, act_p, has_local = dm._find_data_paths()
    assert has_local is True
    assert obs_p == obs_file
    assert act_p == act_file


def test__mtrssm_select_effective_dir_no_candidate(tmp_path: Path) -> None:
    """Test _select_effective_dir when no candidate has required files."""
    class TestConfig(EpisodeDataModuleConfig):
        @property
        def data_root(self) -> Path:
            return tmp_path

        @property
        def processed_data_dir(self) -> Path:
            return tmp_path / "processed_empty"

    config = TestConfig(
        data_name="x",
        batch_size=2,
        num_workers=1,
        gdrive_url="https://example.com",
        observation_file_name="obs",
        observation_preprocess=identity_transform,
        observation_input_transform=identity_transform,
        observation_target_transform=identity_transform,
        action_preprocess=identity_transform,
        action_input_transform=identity_transform,
        action_target_transform=identity_transform,
    )
    effective_dir, uses_vision = EpisodeDataModule._select_effective_dir(
        cast("EpisodeDataModuleConfig", config),
    )
    assert effective_dir == tmp_path / "processed_empty"
    assert uses_vision is False


def test__mtrssm_is_processed_data_ready_no_dir(tmp_path: Path) -> None:
    """Test _is_processed_data_ready when effective_dir does not exist."""
    processed_dir = tmp_path / "nonexistent"

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
        num_workers=1,
        gdrive_url="https://example.com",
        observation_file_name="obs",
        observation_preprocess=identity_transform,
        observation_input_transform=identity_transform,
        observation_target_transform=identity_transform,
        action_preprocess=identity_transform,
        action_input_transform=identity_transform,
        action_target_transform=identity_transform,
    )
    dm = EpisodeDataModule(config)
    assert dm._is_processed_data_ready() is False


def test__mtrssm_setup_stage_test(mtrssm_dataset_config: EpisodeDataModuleConfig) -> None:
    """Test setup with stage='test' does not create train_dataset."""
    dm = EpisodeDataModule(mtrssm_dataset_config)
    dm.setup("test")
    assert dm.train_dataset is None
    assert dm.val_dataset is not None


def test__mtrssm_align_episode_paths_warning(tmp_path: Path) -> None:
    """Test _align_episode_paths with partial id match triggers warning."""
    base = tmp_path / "episodes"
    base.mkdir()
    (base / "act_001.pt").touch()
    (base / "act_002.pt").touch()
    (base / "obs_001.pt").touch()
    action_paths = sorted(base.glob("act_*.pt"))
    observation_paths = sorted(base.glob("obs_*.pt"))

    with pytest.warns(UserWarning, match="Aligning episodes"):
        EpisodeDataModule._align_episode_paths(action_paths, observation_paths)
