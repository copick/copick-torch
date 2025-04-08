#!/usr/bin/env python
"""
Demonstration of the Fourier domain augmentation in copick-torch.

This script shows how to use the FourierAugment3D class in copick-torch 
to augment 3D volumes in the frequency domain.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pathlib import Path

# Add parent directory to path to import copick_torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copick_torch.augmentations import FourierAugment3D, AugmentationFactory


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
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2)
        
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
    grid = ImageGrid(fig, 111, nrows_ncols=(1, num_volumes), axes_pad=0.3, share_all=True,
                   cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.1)
    
    for i, (volume, title) in enumerate(zip(volumes, titles)):
        im = grid[i].imshow(volume[slice_idx], cmap='gray')
        grid[i].set_title(title)
        grid[i].axis('off')
    
    # Add colorbar
    plt.colorbar(im, cax=grid.cbar_axes[0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
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
    fig = plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(magnitude[slice_idx], cmap='viridis')
    plt.title('Original Fourier Spectrum')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(aug_magnitude[slice_idx], cmap='viridis')
    plt.title('Augmented Fourier Spectrum')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def run_multiple_augmentations(volume, num_augmentations=5):
    """Run multiple augmentations on the same volume and visualize them."""
    # Create augmenter
    fourier_aug = FourierAugment3D(
        freq_mask_prob=0.3,
        phase_noise_std=0.1,
        intensity_scaling_range=(0.8, 1.2)
    )
    
    # Run multiple augmentations
    augmented_volumes = [fourier_aug(volume) for _ in range(num_augmentations)]
    
    # Prepare titles
    titles = ['Original'] + [f'Augmentation {i+1}' for i in range(num_augmentations)]
    
    # Visualize
    visualize_volumes([volume] + augmented_volumes, titles, 
                     save_path='fourier_augmentations.png')


def demo_monai_augmentations(volume):
    """Demonstrate using MONAI augmentations via the AugmentationFactory."""
    print("\nDemonstrating MONAI augmentations through AugmentationFactory:")
    
    augmentations = [
        "gaussian_noise",
        "rician_noise",
        "gibbs_noise",
        "gaussian_smooth",
        "gaussian_sharpen",
        "histogram_shift",
        "kspace_spike"
    ]
    
    output_volumes = [volume]  # Original volume first
    titles = ["Original"]
    
    # Apply each augmentation individually for demonstration
    for aug_name in augmentations:
        print(f"Applying {aug_name}...")
        # Create a transform with 100% probability for demonstration
        transform = AugmentationFactory.create_transforms([aug_name], prob=1.0)
        
        # Apply transform - handle both MONAI compose and fallback function
        if hasattr(transform, '__call__'):
            # Simple function (fallback)
            augmented = transform(volume.copy())
        else:
            # MONAI transform - add channel dimension temporarily
            vol_with_channel = volume.copy()[None]  # Add channel dimension
            augmented = transform(vol_with_channel)[0]  # Remove channel dimension
        
        output_volumes.append(augmented)
        titles.append(aug_name.replace('_', ' ').title())
    
    # Visualize in a grid - multiple rows if needed
    max_cols = 4
    rows = (len(output_volumes) + max_cols - 1) // max_cols
    cols = min(max_cols, len(output_volumes))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    slice_idx = volume.shape[0] // 2  # Middle slice
    
    for i, (vol, title) in enumerate(zip(output_volumes, titles)):
        if i < len(axes):
            ax = axes[i]
            ax.imshow(vol[slice_idx], cmap='gray')
            ax.set_title(title)
            ax.axis('off')
    
    # Hide unused subplots
    for i in range(len(output_volumes), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_output/monai_augmentations.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function to demonstrate Fourier augmentation."""
    # Create output directory
    os.makedirs('augmentation_output', exist_ok=True)
    
    # Generate a synthetic volume
    volume = generate_synthetic_volume(size=64, num_spheres=8)
    print(f"Generated volume with shape {volume.shape}")
    
    # Create the Fourier augmenter
    fourier_aug = FourierAugment3D(
        freq_mask_prob=0.3,
        phase_noise_std=0.1,
        intensity_scaling_range=(0.8, 1.2)
    )
    
    # Apply augmentation
    augmented_volume = fourier_aug(volume)
    print(f"Augmented volume with shape {augmented_volume.shape}")
    
    # Visualize volumes
    visualize_volumes(
        [volume, augmented_volume], 
        ['Original', 'Fourier Augmented'],
        save_path='augmentation_output/volume_comparison.png'
    )
    
    # Visualize Fourier spectra
    visualize_fourier_spectrum(
        volume, 
        augmented_volume,
        save_path='augmentation_output/fourier_spectrum_comparison.png'
    )
    
    # Run and visualize multiple augmentations
    run_multiple_augmentations(volume)
    
    # Demonstrate MONAI augmentations
    demo_monai_augmentations(volume)
    
    print("Visualization images saved to the 'augmentation_output' directory.")


if __name__ == "__main__":
    main()
