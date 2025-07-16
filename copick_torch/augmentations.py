"""
Augmentations for 3D volumes based on MONAI transform interface.

This module provides MONAI-based implementations of augmentations for 3D tomographic data.
"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms import (
    Fourier,
    MapTransform,
    RandomizableTrait,
    RandomizableTransform,
    Transform,
)
from monai.transforms.utils import Fourier as FourierUtils
from monai.utils import convert_data_type, convert_to_dst_type, convert_to_tensor


class MixupTransform(RandomizableTransform):
    """
    Implements Mixup augmentation for 3D volumes based on MONAI transform interface.

    Mixup is a data augmentation technique that creates virtual training examples
    by mixing pairs of inputs and their labels with a random proportion.

    Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
    https://arxiv.org/abs/1710.09412
    """

    def __init__(self, alpha: float = 0.2, prob: float = 1.0):
        """
        Initialize the Mixup augmentation.

        Args:
            alpha: Parameter for Beta distribution. Higher values result in more mixing.
            prob: Probability of applying the transform.
        """
        RandomizableTransform.__init__(self, prob)
        self.alpha = alpha
        self.lam = 1.0
        self.index = None

    def randomize(self, data=None) -> None:
        """
        Randomize the transform parameters.
        """
        super().randomize(None)
        if not self._do_transform:
            return None

        if self.alpha > 0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1.0

        # Comment: Previous implementation had a bug that maximized lambda
        # Ensure lambda is between 0 and 1
        self.lam = min(max(self.lam, 0.0), 1.0)

    def __call__(
        self,
        img: torch.Tensor,
        randomize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup augmentation to a batch of images and labels.

        Args:
            img: Tensor of shape [batch_size, channels, depth, height, width]
            randomize: Whether to execute randomize function first, default to True.

        Returns:
            Tuple of (mixed_images, label_a, label_b, lam) where:
                - mixed_images: The mixup result
                - label_a: Original labels
                - label_b: Mixed-in labels
                - lam: Mixing coefficient from Beta distribution
        """
        if randomize:
            self.randomize()

        if not self._do_transform:
            return img, img, img, 1.0

        img = convert_to_tensor(img)
        batch_size = img.shape[0]

        # Generate random indices for mixing
        self.index = torch.randperm(batch_size, device=img.device)

        # Mix the images
        mixed_images = self.lam * img + (1 - self.lam) * img[self.index]

        # Return the mixed images and indices
        return mixed_images, img, img[self.index], self.lam

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


class FourierAugment3D(RandomizableTransform, Fourier):
    """
    Implements Fourier-based augmentation for 3D volumes based on MONAI transform interface.

    This augmentation performs operations in the frequency domain, including
    random frequency dropout (masking), phase noise injection, and intensity scaling.

    It can help the model become more robust to various frequency distortions that
    may occur in tomographic data.
    """

    def __init__(
        self,
        freq_mask_prob: float = 0.3,
        phase_noise_std: float = 0.1,
        intensity_scaling_range: Tuple[float, float] = (0.8, 1.2),
        prob: float = 1.0,
    ) -> None:
        """
        Initialize the Fourier domain augmentation.

        Args:
            freq_mask_prob: Probability of masking a frequency component
            phase_noise_std: Standard deviation of Gaussian noise added to the phase
            intensity_scaling_range: Range for random intensity scaling (min, max)
            prob: Probability of applying the transform
        """
        RandomizableTransform.__init__(self, prob)
        self.freq_mask_prob = freq_mask_prob
        self.phase_noise_std = phase_noise_std
        self.intensity_scaling_range = intensity_scaling_range

        # Randomized parameters
        self._mask = None
        self._phase_noise = None
        self._intensity_scale = None

    def randomize(self, spatial_shape=None) -> None:
        """
        Randomize the transform parameters.
        """
        super().randomize(None)
        if not self._do_transform or spatial_shape is None:
            return None

        # Randomize masking
        if np.random.rand() < self.freq_mask_prob:
            self._mask = torch.rand(spatial_shape, dtype=torch.float32) > self.freq_mask_prob
        else:
            self._mask = None

        # Randomize phase noise
        self._phase_noise = torch.randn(spatial_shape, dtype=torch.float32) * self.phase_noise_std

        # Randomize intensity scaling
        self._intensity_scale = np.random.uniform(
            low=self.intensity_scaling_range[0],
            high=self.intensity_scaling_range[1],
        )

    def __call__(self, volume: torch.Tensor, randomize: bool = True) -> torch.Tensor:
        """
        Apply Fourier domain augmentation to a volume.

        Args:
            volume: Tensor of shape [depth, height, width] or [channels, depth, height, width]
            randomize: Whether to execute randomize function first, default to True.

        Returns:
            Augmented volume with same shape as input
        """
        if randomize:
            # Get input shape for randomization
            input_shape = volume.shape
            spatial_shape = input_shape if len(input_shape) == 3 else input_shape[1:]
            self.randomize(spatial_shape)

        if not self._do_transform:
            return volume

        # Ensure volume is a torch tensor
        volume = convert_to_tensor(volume)
        is_channel_first = len(volume.shape) == 4

        if is_channel_first:
            # Process each channel independently with different random parameters
            # to ensure channel diversity
            output = []
            for channel in range(volume.shape[0]):
                # Re-randomize parameters for each channel to ensure diversity
                if randomize:
                    self.randomize(volume[channel].shape)
                aug_channel = self._apply_fourier_aug(volume[channel])
                output.append(aug_channel)
            return torch.stack(output)
        else:
            # Process 3D volume directly
            return self._apply_fourier_aug(volume)

    def _apply_fourier_aug(self, vol_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier augmentation to a single tensor (no channels).

        Args:
            vol_tensor: 3D tensor of shape [depth, height, width]

        Returns:
            Augmented tensor of same shape
        """
        device = vol_tensor.device

        # Move randomized parameters to the same device
        if self._mask is not None:
            mask = self._mask.to(device)
        phase_noise = self._phase_noise.to(device)

        # FFT
        f_volume = torch.fft.fftn(vol_tensor)
        f_shifted = torch.fft.fftshift(f_volume)

        # Magnitude and phase
        magnitude = torch.abs(f_shifted)
        phase = torch.angle(f_shifted)

        # 1. Random frequency dropout (mask)
        if self._mask is not None:
            magnitude = magnitude * mask

        # 2. Random phase noise
        phase = phase + phase_noise

        # 3. Combine magnitude and noisy phase
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        f_augmented = torch.complex(real, imag)

        # IFFT
        f_ishifted = torch.fft.ifftshift(f_augmented)
        augmented_volume = torch.fft.ifftn(f_ishifted).real  # Discard imaginary part

        # 4. Intensity scaling
        augmented_volume *= self._intensity_scale

        return augmented_volume
