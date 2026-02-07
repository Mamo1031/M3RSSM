"""Tests of `transform.py`."""

import pytest
import torch

from models.transform import (
    GaussianNoise,
    NormalizeAudioMelSpectrogram,
    NormalizeVisionImage,
    RemoveDim,
    ResizeSpatial,
)


def test__remove_dim() -> None:
    """Test `RemoveDim` transform."""
    axis = 1
    indices_to_remove = [0, 2]
    transform = RemoveDim(axis=axis, indices_to_remove=indices_to_remove)

    data = torch.rand(3, 5)
    removed = transform(data)
    assert removed.shape == (3, 3)

    data_3d = torch.rand(2, 4, 3)
    transform_3d = RemoveDim(axis=1, indices_to_remove=[1, 3])
    removed_3d = transform_3d(data_3d)
    assert removed_3d.shape == (2, 2, 3)


def test__gaussian_noise() -> None:
    """Test `GaussianNoise` transform."""
    std = 0.05
    transform = GaussianNoise(std=std)

    data = torch.rand(3, 4)
    noisy = transform(data)
    assert noisy.shape == data.shape
    assert not torch.allclose(data, noisy, atol=1e-6)

    transform_zero = GaussianNoise(std=0.0)
    noisy_zero = transform_zero(data)
    assert torch.allclose(data, noisy_zero, atol=1e-6)


def test__normalize_vision_image() -> None:
    """Test `NormalizeVisionImage` transform."""
    transform = NormalizeVisionImage()

    image = torch.tensor([[[0.0, 127.5, 255.0]]])
    normalized = transform(image)
    assert normalized.shape == image.shape
    expected = torch.tensor([[[-1.0, 0.0, 1.0]]])
    assert torch.allclose(normalized, expected, atol=1e-6)

    batch_image = torch.rand(2, 3, 64, 64) * 255.0
    normalized_batch = transform(batch_image)
    assert normalized_batch.shape == batch_image.shape
    assert normalized_batch.min() >= -1.0
    assert normalized_batch.max() <= 1.0


def test__normalize_audio_mel_spectrogram() -> None:
    """Test `NormalizeAudioMelSpectrogram` transform."""
    min_value = -80.0
    max_value = 0.1
    transform = NormalizeAudioMelSpectrogram(min_value=min_value, max_value=max_value)

    mel_spec = torch.tensor([[[-80.0, -40.0, 0.0, 0.1]]])
    normalized = transform(mel_spec)
    assert normalized.shape == mel_spec.shape
    assert torch.allclose(normalized[0, 0, 0], torch.tensor(-1.0), atol=1e-3)
    assert torch.allclose(normalized[0, 0, 3], torch.tensor(1.0), atol=1e-3)

    batch_mel = torch.rand(2, 80, 64) * (max_value - min_value) + min_value
    normalized_batch = transform(batch_mel)
    assert normalized_batch.shape == batch_mel.shape
    assert normalized_batch.min() >= -1.0
    assert normalized_batch.max() <= 1.0


def test__resize_spatial_3d() -> None:
    """Test `ResizeSpatial` transform with 3D tensor (c, h, w)."""
    transform = ResizeSpatial(height=32, width=32)
    data = torch.rand(3, 64, 64)
    resized = transform(data)
    assert resized.shape == (3, 32, 32)


def test__resize_spatial_4d() -> None:
    """Test `ResizeSpatial` transform with 4D tensor (t, c, h, w)."""
    transform = ResizeSpatial(height=32, width=32)
    data = torch.rand(4, 3, 64, 64)
    resized = transform(data)
    assert resized.shape == (4, 3, 32, 32)


def test__resize_spatial_5d() -> None:
    """Test `ResizeSpatial` transform with 5D tensor (batch, time, c, h, w)."""
    transform = ResizeSpatial(height=32, width=32)
    data = torch.rand(2, 4, 3, 64, 64)
    resized = transform(data)
    assert resized.shape == (2, 4, 3, 32, 32)


def test__resize_spatial_unsupported() -> None:
    """Test `ResizeSpatial` with unsupported tensor shape."""
    transform = ResizeSpatial(height=32, width=32)
    data = torch.rand(3, 4)
    with pytest.raises(ValueError, match="Unsupported"):
        transform(data)
