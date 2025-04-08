import random
import torch
import numpy as np
import torch.fft

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
