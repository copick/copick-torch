import torch
import numpy as np
import pytest
from copick_torch.augmentations import MixupAugmentation, FourierAugment3D


def test_mixup_augmentation():
    """Test MixupAugmentation on toy data."""
    # Create sample data
    batch_size = 4
    channels = 1
    depth = 8
    height = 8
    width = 8
    images = torch.randn((batch_size, channels, depth, height, width))
    labels = torch.tensor([0, 1, 2, 3])
    
    # Apply mixup
    mixup = MixupAugmentation(alpha=0.2)
    mixed_images, labels_a, labels_b, lam = mixup(images, labels)
    
    # Check output shapes
    assert mixed_images.shape == images.shape
    assert labels_a.shape == labels.shape
    assert labels_b.shape == labels.shape
    assert isinstance(lam, float)
    assert 0.0 <= lam <= 1.0
    
    # Check mixup criterion
    criterion = torch.nn.CrossEntropyLoss()
    preds = torch.randn((batch_size, 4))  # 4 classes
    loss = MixupAugmentation.mixup_criterion(criterion, preds, labels_a, labels_b, lam)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar


def test_fourier_augmentation_with_tensor():
    """Test FourierAugment3D with PyTorch tensor input."""
    # Create a test volume
    volume = torch.randn((16, 16, 16))
    
    # Create augmenter
    fourier_aug = FourierAugment3D(
        freq_mask_prob=0.3,
        phase_noise_std=0.1,
        intensity_scaling_range=(0.8, 1.2)
    )
    
    # Apply augmentation
    augmented_volume = fourier_aug(volume)
    
    # Check output shape and type
    assert augmented_volume.shape == volume.shape
    assert isinstance(augmented_volume, torch.Tensor)
    
    # Check that the augmented volume is different from the original
    assert not torch.allclose(volume, augmented_volume)


def test_fourier_augmentation_with_numpy():
    """Test FourierAugment3D with NumPy array input."""
    # Create a test volume
    volume = np.random.randn(16, 16, 16).astype(np.float32)
    
    # Create augmenter
    fourier_aug = FourierAugment3D(
        freq_mask_prob=0.3,
        phase_noise_std=0.1,
        intensity_scaling_range=(0.8, 1.2)
    )
    
    # Apply augmentation
    augmented_volume = fourier_aug(volume)
    
    # Check output shape and type
    assert augmented_volume.shape == volume.shape
    assert isinstance(augmented_volume, np.ndarray)
    
    # Check that the augmented volume is different from the original
    assert not np.allclose(volume, augmented_volume)


def test_fourier_augmentation_params():
    """Test FourierAugment3D with different parameters."""
    # Create a test volume
    volume = torch.randn((16, 16, 16))
    
    # Test with higher frequency mask probability
    high_freq_mask = FourierAugment3D(
        freq_mask_prob=1.0,  # Always mask
        phase_noise_std=0.0,  # No phase noise
        intensity_scaling_range=(1.0, 1.0)  # No intensity scaling
    )
    high_freq_augmented = high_freq_mask(volume)
    
    # Test with higher phase noise
    high_phase_noise = FourierAugment3D(
        freq_mask_prob=0.0,  # No frequency masking
        phase_noise_std=0.5,  # High phase noise
        intensity_scaling_range=(1.0, 1.0)  # No intensity scaling
    )
    high_phase_augmented = high_phase_noise(volume)
    
    # Test with intensity scaling only
    intensity_scaling = FourierAugment3D(
        freq_mask_prob=0.0,  # No frequency masking
        phase_noise_std=0.0,  # No phase noise
        intensity_scaling_range=(0.5, 2.0)  # Higher intensity scaling range
    )
    intensity_augmented = intensity_scaling(volume)
    
    # Each augmentation should produce a different result
    assert not torch.allclose(high_freq_augmented, high_phase_augmented)
    assert not torch.allclose(high_freq_augmented, intensity_augmented)
    assert not torch.allclose(high_phase_augmented, intensity_augmented)


def test_fourier_augmentation_dimensions():
    """Test FourierAugment3D with different input dimensions."""
    # Create augmenter
    fourier_aug = FourierAugment3D(
        freq_mask_prob=0.3,
        phase_noise_std=0.1,
        intensity_scaling_range=(0.8, 1.2)
    )
    
    # Test with different volume shapes
    for shape in [(8, 8, 8), (16, 16, 16), (32, 32, 32), (16, 24, 32)]:
        volume = torch.randn(shape)
        augmented_volume = fourier_aug(volume)
        
        # Check output shape matches input shape
        assert augmented_volume.shape == volume.shape


def test_fourier_augmentation_error_handling():
    """Test FourierAugment3D error handling."""
    # Create augmenter
    fourier_aug = FourierAugment3D(
        freq_mask_prob=0.3,
        phase_noise_std=0.1,
        intensity_scaling_range=(0.8, 1.2)
    )
    
    # Test with 2D input (should raise assertion error)
    with pytest.raises(AssertionError):
        volume_2d = torch.randn((16, 16))
        fourier_aug(volume_2d)
    
    # Test with 4D input (should raise assertion error)
    with pytest.raises(AssertionError):
        volume_4d = torch.randn((1, 16, 16, 16))
        fourier_aug(volume_4d)


def test_fourier_augmentation_reproducibility():
    """Test that FourierAugment3D produces reproducible results with the same seed."""
    # Create a test volume
    volume = torch.randn((16, 16, 16))
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create augmenter
    fourier_aug = FourierAugment3D(
        freq_mask_prob=0.3,
        phase_noise_std=0.1,
        intensity_scaling_range=(0.8, 1.2)
    )
    
    # Apply augmentation twice with the same seed
    torch.manual_seed(42)
    np.random.seed(42)
    augmented_1 = fourier_aug(volume.clone())
    
    torch.manual_seed(42)
    np.random.seed(42)
    augmented_2 = fourier_aug(volume.clone())
    
    # Results should be identical
    assert torch.allclose(augmented_1, augmented_2)
    
    # With a different seed, results should be different
    torch.manual_seed(0)
    np.random.seed(0)
    augmented_3 = fourier_aug(volume.clone())
    assert not torch.allclose(augmented_1, augmented_3)
