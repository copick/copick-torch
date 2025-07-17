#!/usr/bin/env python
"""
Demonstration of the Fourier domain augmentation in copick-torch.

This script shows how to use the FourierAugment3D class in copick-torch
to augment 3D volumes in the frequency domain.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import ImageGrid

# Add parent directory to path to import copick_torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copick_torch.augmentations import FourierAugment3D


def generate_synthetic_volume(size=32, num_spheres=5):
    """Generate a synthetic volume with random spheres for demonstration."""
    volume = np.zeros((size, size, size), dtype=np.float32)

    # Add random spheres
    for _ in range(num_spheres):
        radius = np.random.randint(3, 8)
        center_x = np.random.randint(radius, size - radius)
        center_y = np.random.randint(radius, size - radius)
        center_z = np.random.randint(radius, size - radius)

        # Create coordinate grids
        z, y, x = np.ogrid[:size, :size, :size]

        # Calculate distance from center
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2 + (z - center_z) ** 2)

        # Add sphere
        sphere = (dist <= radius) * np.random.uniform(0.5, 1.0)
        volume += sphere

    # Normalize volume
    volume = (volume - np.mean(volume)) / np.std(volume)

    return volume


def visualize_volumes(volumes, titles, slice_idx=None, save_path=None):
    """Visualize central slices of multiple volumes."""
    num_volumes = len(volumes)

    # Determine slice index for visualization if not provided
    if slice_idx is None:
        slice_idx = volumes[0].shape[0] // 2

    fig = plt.figure(figsize=(4 * num_volumes, 4))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(1, num_volumes),
        axes_pad=0.3,
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="5%",
        cbar_pad=0.1,
    )

    for i, (volume, title) in enumerate(zip(volumes, titles)):
        im = grid[i].imshow(volume[slice_idx], cmap="gray")
        grid[i].set_title(title)
        grid[i].axis("off")

    # Add colorbar
    plt.colorbar(im, cax=grid.cbar_axes[0])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def visualize_fourier_spectrum(volume, augmented_volume, slice_idx=None, save_path=None):
    """Visualize original and augmented Fourier spectrum."""
    # Compute Fourier transforms
    f_volume = np.fft.fftn(volume)
    f_volume_shifted = np.fft.fftshift(f_volume)
    f_aug_volume = np.fft.fftn(augmented_volume)
    f_aug_volume_shifted = np.fft.fftshift(f_aug_volume)

    # Convert to magnitude (log scale for better visualization)
    magnitude = np.log(np.abs(f_volume_shifted) + 1)
    aug_magnitude = np.log(np.abs(f_aug_volume_shifted) + 1)

    # Determine slice index for visualization if not provided
    if slice_idx is None:
        slice_idx = volume.shape[0] // 2

    # Create visualization
    _ = plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(magnitude[slice_idx], cmap="viridis")
    plt.title("Original Fourier Spectrum")
    plt.colorbar()
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(aug_magnitude[slice_idx], cmap="viridis")
    plt.title("Augmented Fourier Spectrum")
    plt.colorbar()
    plt.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def run_multiple_augmentations(volume, num_augmentations=5):
    """Run multiple augmentations on the same volume and visualize them."""
    # Create augmenter
    fourier_aug = FourierAugment3D(freq_mask_prob=0.3, phase_noise_std=0.1, intensity_scaling_range=(0.8, 1.2))

    # Run multiple augmentations
    augmented_volumes = []
    for _ in range(num_augmentations):
        augmented_volumes.append(fourier_aug(torch.from_numpy(volume), randomize=True).numpy())

    # Prepare titles
    titles = ["Original"] + [f"Augmentation {i+1}" for i in range(num_augmentations)]

    # Visualize
    visualize_volumes([volume] + augmented_volumes, titles, save_path="fourier_augmentations.png")


def demonstrate_fourier_augmentation():
    """Demonstrate the MONAI-based implementation."""
    # Generate synthetic volume
    volume = generate_synthetic_volume(size=64, num_spheres=8)
    volume_tensor = torch.from_numpy(volume)

    # Create the augmenter with defined parameters
    fourier_aug = FourierAugment3D(freq_mask_prob=0.3, phase_noise_std=0.1, intensity_scaling_range=(0.8, 1.2))

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Apply augmentation
    augmented = fourier_aug(volume_tensor, randomize=True).numpy()

    # Visualize comparison
    titles = ["Original", "Fourier Augmentation"]
    visualize_volumes([volume, augmented], titles, save_path="fourier_demo.png")


def main():
    """Main function to demonstrate Fourier augmentation."""
    # Create output directory
    os.makedirs("augmentation_output", exist_ok=True)

    # Generate a synthetic volume
    volume = generate_synthetic_volume(size=64, num_spheres=8)
    print(f"Generated volume with shape {volume.shape}")

    # Create the Fourier augmenter
    fourier_aug = FourierAugment3D(freq_mask_prob=0.3, phase_noise_std=0.1, intensity_scaling_range=(0.8, 1.2))

    # Convert to tensor for MONAI transform
    volume_tensor = torch.from_numpy(volume)

    # Apply augmentation
    augmented_volume = fourier_aug(volume_tensor).numpy()
    print(f"Augmented volume with shape {augmented_volume.shape}")

    # Visualize volumes
    visualize_volumes(
        [volume, augmented_volume],
        ["Original", "Fourier Augmented"],
        save_path="augmentation_output/volume_comparison.png",
    )

    # Visualize Fourier spectra
    visualize_fourier_spectrum(
        volume,
        augmented_volume,
        save_path="augmentation_output/fourier_spectrum_comparison.png",
    )

    # Run and visualize multiple augmentations
    run_multiple_augmentations(volume)

    # Demonstrate Fourier augmentation with visualization
    demonstrate_fourier_augmentation()

    print("Visualization images saved to the 'augmentation_output' directory.")


if __name__ == "__main__":
    main()
