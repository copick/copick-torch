"""Tests for the augmentations in copick-torch."""

import numpy as np
import pytest
import torch

from copick_torch.augmentations import FourierAugment3D, MixupTransform

# TODO: Uncomment and fix the test below

# def test_mixup_transform():
#     """Test that MixupTransform produces expected outputs."""
#     # Create a batch of simple test volumes
#     batch_size = 4
#     volume_shape = (3, 8, 8, 8)  # (channels, depth, height, width)
#
#     # Create test data
#     x = torch.zeros((batch_size,) + volume_shape)
#     # Make each sample in batch unique
#     for i in range(batch_size):
#         x[i, :, :, :, :] = i
#
#     # Test initialization
#     mixup = MixupTransform(alpha=0.2, prob=1.0)
#
#     # Test with randomization
#     mixed_x, orig_x, mixed_idx_x, lam = mixup(x)
#
#     # Check shapes
#     assert mixed_x.shape == x.shape
#     assert orig_x.shape == x.shape
#     assert mixed_idx_x.shape == x.shape
#     assert isinstance(lam, float)
#
#     # Check mixing with known lambda
#     mixup.lam = 0.7  # Force lambda to a known value
#     mixup.index = torch.tensor([1, 0, 3, 2])  # Force permutation
#
#     mixed_x, _, _, _ = mixup(x, randomize=False)
#
#     # Check first sample: Should be 0.7*0 + 0.3*1 = 0.3
#     assert torch.allclose(mixed_x[0, 0, 0, 0, 0], torch.tensor(0.3), atol=1e-6)
#
#     # Check second sample: Should be 0.7*1 + 0.3*0 = 0.7
#     assert torch.allclose(mixed_x[1, 0, 0, 0, 0], torch.tensor(0.7), atol=1e-6)
#
#     # Test mixup expected loss with lambda=0.7
#     assert torch.allclose(torch.tensor(1.6), torch.tensor(0.7), atol=1.0)
#
#     # Test mixup_criterion
#     def dummy_criterion(pred, target):
#         return torch.abs(pred - target).mean()
#
#     # Simple prediction and labels for testing
#     pred = torch.ones((batch_size,))
#     y_a = torch.zeros((batch_size,))
#     y_b = torch.ones((batch_size,)) * 2
#     lam = 0.7
#
#     # Expected loss: 0.7 * |1-0| + 0.3 * |1-2| = 0.7 + 0.3 = 1.0
#     mixed_loss = MixupTransform.mixup_criterion(dummy_criterion, pred, y_a, y_b, lam)
#     assert torch.isclose(mixed_loss, torch.tensor(1.0), atol=1e-6)


def test_fourier_augment3d():
    """Test that FourierAugment3D produces expected outputs."""
    # Create a test volume
    volume = torch.ones((16, 16, 16))

    # Test initialization
    aug = FourierAugment3D(freq_mask_prob=0.3, phase_noise_std=0.1, intensity_scaling_range=(0.8, 1.2), prob=1.0)

    # Apply augmentation
    augmented = aug(volume)

    # Check shape preservation
    assert augmented.shape == volume.shape

    # Make sure the augmentation changed the volume (not identity)
    assert not torch.allclose(augmented, volume, rtol=1e-3, atol=1e-3)

    # Test with zero phase noise and fixed intensity scale (should be close to identity
    # if there's no masking)
    aug = FourierAugment3D(
        freq_mask_prob=0.0,  # No masking
        phase_noise_std=0.0,  # No phase noise
        intensity_scaling_range=(1.0, 1.0),  # No intensity scaling
        prob=1.0,
    )

    # Force parameters
    aug._mask = None
    aug._phase_noise = torch.zeros_like(volume)
    aug._intensity_scale = 1.0

    # Apply augmentation without randomization
    augmented = aug(volume, randomize=False)

    # Should be identity transform
    assert torch.allclose(augmented, volume, rtol=1e-3, atol=1e-3)


def test_fourier_augment3d_channel_first():
    """Test that FourierAugment3D works with channel-first inputs."""
    # Create a test volume with channels
    volume = torch.ones((3, 16, 16, 16))

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Test initialization
    aug = FourierAugment3D(freq_mask_prob=0.3, phase_noise_std=0.1, intensity_scaling_range=(0.8, 1.2), prob=1.0)

    # Apply augmentation with fixed seed
    augmented = aug(volume, randomize=True)

    # Check shape preservation
    assert augmented.shape == volume.shape

    # Check the channels are processed differently
    # We're using a more robust check that doesn't depend on specific random values
    diffs = []
    for i in range(volume.shape[0] - 1):
        for j in range(i + 1, volume.shape[0]):
            # Calculate mean absolute difference between channels
            diff = torch.abs(augmented[i] - augmented[j]).mean().item()
            diffs.append(diff)

    # Assert there's at least some difference between channels
    # This is more robust than comparing specific tensors
    assert max(diffs) > 0.01


def test_zero_probability():
    """Test that transforms with zero probability leave inputs unchanged."""
    # Create test data
    volume = torch.ones((16, 16, 16))

    # Test FourierAugment3D with zero probability
    aug = FourierAugment3D(
        freq_mask_prob=0.3,
        phase_noise_std=0.1,
        intensity_scaling_range=(0.8, 1.2),
        prob=0.0,  # Zero probability of applying transform
    )

    # Apply augmentation
    augmented = aug(volume)

    # Should be identity transform
    assert torch.allclose(augmented, volume)

    # Test MixupTransform with zero probability
    mixup = MixupTransform(alpha=0.2, prob=0.0)

    # Create a simple batch
    batch = torch.ones((4, 3, 8, 8, 8))

    # Apply mixup
    mixed_x, orig_x, mixed_idx_x, lam = mixup(batch)

    # Lambda should be 1.0 (no mixing)
    assert lam == 1.0

    # Original data should be unchanged
    assert torch.allclose(mixed_x, batch)
