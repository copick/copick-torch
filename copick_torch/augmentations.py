"""
Augmentation functions for copick-torch.

This module provides augmentation functions for 3D volumes, particularly for cryo-electron tomography data.
It uses MONAI transforms for common operations and extends with custom transformations
specific to copick's needs.
"""

import random
import torch
import numpy as np
import torch.fft
from typing import Tuple, Union, Optional, List, Dict, Any, Sequence

try:
    # Import from MONAI if available
    from monai.transforms import (
        RandGaussianNoise, 
        RandRicianNoise, 
        RandScaleIntensity, 
        NormalizeIntensity,
        RandBiasField,
        RandAdjustContrast,
        RandGaussianSmooth,
        RandGaussianSharpen,
        RandHistogramShift,
        RandGibbsNoise,
        GibbsNoise,
        KSpaceSpikeNoise,
        RandKSpaceSpikeNoise,
        SpatialCrop,
        Compose
    )
    from monai.transforms.utils import Fourier
    HAS_MONAI = True
except ImportError:
    # Fallback to custom implementations
    HAS_MONAI = False
    # Let user know MONAI is recommended
    import warnings
    warnings.warn(
        "MONAI is not installed. Using copick's custom implementations. "
        "For optimal performance, consider installing MONAI: pip install monai"
    )


class MixupAugmentation:
    """
    Implements Mixup augmentation for 3D volumes.
    
    Mixup is a data augmentation technique that creates virtual training examples
    by mixing pairs of inputs and their labels with a random proportion.
    
    Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
    https://arxiv.org/abs/1710.09412
    """
    
    def __init__(self, alpha=0.2):
        """
        Initialize the Mixup augmentation.
        
        Args:
            alpha: Parameter for Beta distribution. Higher values result in more mixing.
        """
        self.alpha = alpha
        
    def __call__(self, images, labels):
        """
        Apply mixup augmentation to a batch of images and labels.
        
        Args:
            images: Tensor of shape [batch_size, channels, depth, height, width]
            labels: Tensor of shape [batch_size]
            
        Returns:
            Tuple of (mixed_images, label_a, label_b, lam) where:
                - mixed_images: The mixup result
                - label_a: Original labels
                - label_b: Mixed-in labels
                - lam: Mixing coefficient from Beta distribution
        """
        batch_size = images.size(0)
        
        # Sample mixing coefficient from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
            
        # Ensure lam is within reasonable bounds
        lam = max(lam, 1 - lam)
        
        # Generate random indices for mixing
        index = torch.randperm(batch_size, device=images.device)
        
        # Mix the images
        mixed_images = lam * images + (1 - lam) * images[index]
        
        # Return the mixed images and label information
        return mixed_images, labels, labels[index], lam
    
    @staticmethod
    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        """
        Apply mixup to the loss calculation.
        
        Args:
            criterion: Loss function
            pred: Model predictions
            y_a: First labels
            y_b: Second (mixed-in) labels
            lam: Mixing coefficient
            
        Returns:
            Mixed loss
        """
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class FourierAugment3D:
    """
    Implements Fourier-based augmentation for 3D volumes.
    
    This augmentation performs operations in the frequency domain, including
    random frequency dropout (masking), phase noise injection, and intensity scaling.
    
    It can help the model become more robust to various frequency distortions that
    may occur in tomographic data.
    """
    
    def __init__(self, freq_mask_prob=0.3, phase_noise_std=0.1, intensity_scaling_range=(0.8, 1.2)):
        """
        Initialize the Fourier domain augmentation.
        
        Args:
            freq_mask_prob: Probability of masking a frequency component
            phase_noise_std: Standard deviation of Gaussian noise added to the phase
            intensity_scaling_range: Range for random intensity scaling (min, max)
        """
        self.freq_mask_prob = freq_mask_prob
        self.phase_noise_std = phase_noise_std
        self.intensity_scaling_range = intensity_scaling_range
        
        # If MONAI is available, we can use its functions to implement some operations
        if HAS_MONAI:
            self.fourier_util = Fourier()
        
    def __call__(self, volume):
        """
        Apply Fourier domain augmentation to a volume.
        
        Args:
            volume: Tensor of shape [depth, height, width] or numpy array
            
        Returns:
            Augmented volume with same shape as input
        """
        # Ensure volume is a torch tensor
        is_numpy = isinstance(volume, np.ndarray)
        if is_numpy:
            volume = torch.from_numpy(volume.copy()).float()
        else:
            volume = volume.clone()
        
        # Ensure 3D volume
        assert volume.ndim == 3, f"Expected 3D volume, got shape {volume.shape}"
        
        # FFT
        f_volume = torch.fft.fftn(volume)
        f_shifted = torch.fft.fftshift(f_volume)
        
        # Magnitude and phase
        magnitude = torch.abs(f_shifted)
        phase = torch.angle(f_shifted)
        
        # 1. Random frequency dropout (mask)
        if torch.rand(1).item() < self.freq_mask_prob:
            mask = torch.rand_like(magnitude) > self.freq_mask_prob
            magnitude = magnitude * mask
        
        # 2. Random phase noise
        phase = phase + torch.randn_like(phase) * self.phase_noise_std
        
        # 3. Combine magnitude and noisy phase
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        f_augmented = torch.complex(real, imag)
        
        # IFFT
        f_ishifted = torch.fft.ifftshift(f_augmented)
        augmented_volume = torch.fft.ifftn(f_ishifted).real  # Discard imaginary part
        
        # 4. Intensity scaling
        scale = torch.empty(1).uniform_(*self.intensity_scaling_range).item()
        augmented_volume *= scale
        
        # Convert back to numpy if input was numpy
        if is_numpy:
            augmented_volume = augmented_volume.numpy()
        
        return augmented_volume


class AugmentationFactory:
    """
    Factory class for creating augmentation transforms.
    
    This class provides methods to create various augmentation transforms,
    using MONAI implementations when available and falling back to custom
    implementations otherwise.
    """
    
    @staticmethod
    def create_transforms(augmentation_types, prob=0.5):
        """
        Create a composition of transforms based on the specified augmentation types.
        
        Args:
            augmentation_types: List of augmentation types to include
            prob: Base probability for each transform
            
        Returns:
            Compose transform that applies the requested augmentations
        """
        transforms = []
        
        for aug_type in augmentation_types:
            if aug_type == "gaussian_noise":
                transforms.append(AugmentationFactory.create_gaussian_noise(prob))
            elif aug_type == "rician_noise":
                transforms.append(AugmentationFactory.create_rician_noise(prob))
            elif aug_type == "gibbs_noise":
                transforms.append(AugmentationFactory.create_gibbs_noise(prob))
            elif aug_type == "scale_intensity":
                transforms.append(AugmentationFactory.create_scale_intensity(prob))
            elif aug_type == "adjust_contrast":
                transforms.append(AugmentationFactory.create_adjust_contrast(prob))
            elif aug_type == "gaussian_smooth":
                transforms.append(AugmentationFactory.create_gaussian_smooth(prob))
            elif aug_type == "gaussian_sharpen":
                transforms.append(AugmentationFactory.create_gaussian_sharpen(prob))
            elif aug_type == "histogram_shift":
                transforms.append(AugmentationFactory.create_histogram_shift(prob))
            elif aug_type == "kspace_spike":
                transforms.append(AugmentationFactory.create_kspace_spike(prob))
            elif aug_type == "fourier":
                transforms.append(FourierAugment3D())  # No direct MONAI equivalent
        
        if HAS_MONAI:
            return Compose(transforms)
        else:
            # Simple composition function if MONAI is not available
            def compose_fn(x):
                for transform in transforms:
                    x = transform(x)
                return x
            return compose_fn
    
    @staticmethod
    def create_gaussian_noise(prob=0.1, mean=0.0, std=0.1):
        """Create a Gaussian noise transform."""
        if HAS_MONAI:
            return RandGaussianNoise(prob=prob, mean=mean, std=std)
        else:
            # Fallback implementation
            def gaussian_noise(x):
                if random.random() < prob:
                    if isinstance(x, np.ndarray):
                        noise = np.random.normal(mean, std, size=x.shape)
                        return x + noise
                    else:  # Assume torch.Tensor
                        noise = torch.randn_like(x) * std + mean
                        return x + noise
                return x
            return gaussian_noise
    
    @staticmethod
    def create_rician_noise(prob=0.1, std=0.1):
        """Create a Rician noise transform."""
        if HAS_MONAI:
            return RandRicianNoise(prob=prob, mean=0.0, std=std)
        else:
            # Simplified fallback implementation
            def rician_noise(x):
                if random.random() < prob:
                    if isinstance(x, np.ndarray):
                        noise1 = np.random.normal(0, std, size=x.shape)
                        noise2 = np.random.normal(0, std, size=x.shape)
                        return np.sqrt((x + noise1)**2 + noise2**2)
                    else:  # Assume torch.Tensor
                        noise1 = torch.randn_like(x) * std
                        noise2 = torch.randn_like(x) * std
                        return torch.sqrt((x + noise1)**2 + noise2**2)
                return x
            return rician_noise
    
    @staticmethod
    def create_gibbs_noise(prob=0.1, alpha=0.3):
        """Create a Gibbs noise transform."""
        if HAS_MONAI:
            return RandGibbsNoise(prob=prob, alpha=alpha)
        else:
            # Simplified fallback - this is complex to implement from scratch
            # so we just return a pass-through function and warn the user
            warnings.warn(
                "Gibbs noise requires MONAI. This transform will be skipped."
            )
            return lambda x: x
    
    @staticmethod
    def create_scale_intensity(prob=0.1, factors=(-0.2, 0.2)):
        """Create a scale intensity transform."""
        if HAS_MONAI:
            return RandScaleIntensity(prob=prob, factors=factors)
        else:
            # Fallback implementation
            def scale_intensity(x):
                if random.random() < prob:
                    factor = random.uniform(factors[0], factors[1])
                    return x * (1 + factor)
                return x
            return scale_intensity
    
    @staticmethod
    def create_adjust_contrast(prob=0.1, gamma=(0.5, 4.5)):
        """Create a contrast adjustment transform."""
        if HAS_MONAI:
            return RandAdjustContrast(prob=prob, gamma=gamma)
        else:
            # Fallback implementation
            def adjust_contrast(x):
                if random.random() < prob:
                    gamma_val = random.uniform(gamma[0], gamma[1])
                    epsilon = 1e-7
                    img_min = x.min()
                    img_range = x.max() - img_min
                    return ((x - img_min) / float(img_range + epsilon)) ** gamma_val * img_range + img_min
                return x
            return adjust_contrast
    
    @staticmethod
    def create_gaussian_smooth(prob=0.1, sigma_range=(0.5, 1.5)):
        """Create a Gaussian smoothing transform."""
        if HAS_MONAI:
            return RandGaussianSmooth(
                prob=prob, 
                sigma_x=sigma_range,
                sigma_y=sigma_range,
                sigma_z=sigma_range
            )
        else:
            # Simplified fallback - would need scipy.ndimage in practice
            warnings.warn(
                "Gaussian smoothing without MONAI requires scipy.ndimage. "
                "This transform might not work correctly."
            )
            def gaussian_smooth(x):
                if random.random() < prob:
                    try:
                        from scipy.ndimage import gaussian_filter
                        sigma = random.uniform(sigma_range[0], sigma_range[1])
                        if isinstance(x, np.ndarray):
                            return gaussian_filter(x, sigma=sigma)
                        else:  # Assume torch.Tensor
                            x_np = x.cpu().numpy()
                            smoothed = gaussian_filter(x_np, sigma=sigma)
                            return torch.from_numpy(smoothed).to(x.device)
                    except ImportError:
                        warnings.warn("scipy.ndimage is required for gaussian_filter")
                        return x
                return x
            return gaussian_smooth
    
    @staticmethod
    def create_gaussian_sharpen(prob=0.1, sigma1_range=(0.5, 1.0), sigma2_range=(0.5, 1.0), alpha_range=(10, 30)):
        """Create a Gaussian sharpening transform."""
        if HAS_MONAI:
            return RandGaussianSharpen(
                prob=prob,
                sigma1_x=sigma1_range,
                sigma1_y=sigma1_range,
                sigma1_z=sigma1_range,
                sigma2_x=sigma2_range,
                sigma2_y=sigma2_range,
                sigma2_z=sigma2_range,
                alpha=alpha_range
            )
        else:
            # This is complex to implement from scratch correctly
            warnings.warn(
                "Gaussian sharpening requires MONAI. This transform will be skipped."
            )
            return lambda x: x
    
    @staticmethod
    def create_histogram_shift(prob=0.1, num_control_points=(5, 10)):
        """Create a histogram shift transform."""
        if HAS_MONAI:
            return RandHistogramShift(prob=prob, num_control_points=num_control_points)
        else:
            # This is complex to implement from scratch correctly
            warnings.warn(
                "Histogram shift requires MONAI. This transform will be skipped."
            )
            return lambda x: x
    
    @staticmethod
    def create_kspace_spike(prob=0.1):
        """Create a k-space spike noise transform."""
        if HAS_MONAI:
            return RandKSpaceSpikeNoise(prob=prob)
        else:
            # This is complex to implement from scratch correctly
            warnings.warn(
                "K-space spike noise requires MONAI. This transform will be skipped."
            )
            return lambda x: x
