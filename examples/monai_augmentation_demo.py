#!/usr/bin/env python
"""
Demonstration of the MONAI-based augmentations in copick-torch.

This script shows how to use the AugmentationComposer class in copick-torch
to apply a combination of intensity and spatial augmentations to 3D volumes.
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

from copick_torch.augmentations import AugmentationComposer, FourierAugment3D


def generate_synthetic_volume(size=32, num_spheres=5, channels=1):
    """Generate a synthetic volume with random spheres for demonstration."""
    if channels > 1:
        volume = np.zeros((channels, size, size, size), dtype=np.float32)
        
        for c in range(channels):
            channel_vol = np.zeros((size, size, size), dtype=np.float32)
            
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
                channel_vol += sphere
            
            # Normalize volume
            channel_vol = (channel_vol - np.mean(channel_vol)) / np.std(channel_vol)
            volume[c] = channel_vol
    else:
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


def visualize_volumes(volumes, titles, slice_idx=None, axis=0, save_path=None):
    """Visualize central slices of multiple volumes."""
    num_volumes = len(volumes)
    
    # Convert volumes to numpy arrays
    volumes_np = []
    for vol in volumes:
        if isinstance(vol, torch.Tensor):
            vol = vol.detach().cpu().numpy()
        volumes_np.append(vol)
    
    # Handle channel dimension for visualization
    for i, vol in enumerate(volumes_np):
        if vol.ndim == 4:  # If the volume has a channel dimension
            # Take the first channel for visualization
            volumes_np[i] = vol[0]
    
    # Determine slice index for visualization if not provided
    if slice_idx is None:
        slice_idx = volumes_np[0].shape[axis] // 2
    
    # Get slices from the volumes
    slices = []
    for vol in volumes_np:
        if axis == 0:
            slices.append(vol[slice_idx])
        elif axis == 1:
            slices.append(vol[:, slice_idx])
        else:  # axis == 2
            slices.append(vol[:, :, slice_idx])
    
    fig = plt.figure(figsize=(4 * num_volumes, 4))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, num_volumes), axes_pad=0.3, share_all=True,
                     cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.1)
    
    for i, (slice_img, title) in enumerate(zip(slices, titles)):
        im = grid[i].imshow(slice_img, cmap='gray')
        grid[i].set_title(title)
        grid[i].axis('off')
    
    # Add colorbar
    plt.colorbar(im, cax=grid.cbar_axes[0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def demonstrate_augmentation_composer():
    """Demonstrate the MONAI-based AugmentationComposer."""
    # Generate synthetic volume with 2 channels
    volume = generate_synthetic_volume(size=64, num_spheres=8, channels=2)
    volume_tensor = torch.from_numpy(volume)
    
    # Create the augmentation composer with various parameters
    augmentation_composer = AugmentationComposer(
        intensity_transforms=True,
        spatial_transforms=True,
        prob_intensity=0.8,
        prob_spatial=0.5,
        rotate_range=0.1,
        scale_range=0.15,
        noise_std=0.1,
        gamma_range=(0.7, 1.3),
        intensity_range=(0.8, 1.2),
        shift_range=(-0.1, 0.1)
    )
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Apply augmentation
    augmented = augmentation_composer(volume_tensor)
    
    # Convert back to numpy for visualization
    augmented_np = augmented.numpy()
    
    # Visualize comparison of first channel
    titles = ['Original', 'MONAI Augmented']
    visualize_volumes([volume[0], augmented_np[0]], titles,
                      save_path='augmentation_output/monai_augmentation_demo.png')


def run_multiple_augmentations(volume, num_augmentations=5):
    """Run multiple augmentations on the same volume and visualize them."""
    # Create augmentation composer
    augmentation_composer = AugmentationComposer(
        intensity_transforms=True,
        spatial_transforms=True,
        prob_intensity=0.7,
        prob_spatial=0.5
    )
    
    # Run multiple augmentations
    augmented_volumes = []
    for _ in range(num_augmentations):
        torch.manual_seed(_ + 42)  # Different seed for each augmentation
        np.random.seed(_ + 42)
        aug_vol = augmentation_composer(torch.from_numpy(volume))
        augmented_volumes.append(aug_vol.numpy())
    
    # Prepare titles
    titles = ['Original'] + [f'Augmentation {i+1}' for i in range(num_augmentations)]
    
    # Visualize
    visualize_volumes([volume] + augmented_volumes, titles,
                      save_path='augmentation_output/multiple_monai_augmentations.png')


def compare_augmentation_types():
    """Compare different types of augmentations."""
    # Generate synthetic volume
    volume = generate_synthetic_volume(size=64, num_spheres=8)
    volume_tensor = torch.from_numpy(volume)
    
    # Create different types of augmenters
    intensity_augmenter = AugmentationComposer(
        intensity_transforms=True,
        spatial_transforms=False,
        prob_intensity=1.0
    )
    
    spatial_augmenter = AugmentationComposer(
        intensity_transforms=False,
        spatial_transforms=True,
        prob_spatial=1.0
    )
    
    combined_augmenter = AugmentationComposer(
        intensity_transforms=True,
        spatial_transforms=True,
        prob_intensity=1.0,
        prob_spatial=1.0
    )
    
    fourier_augmenter = FourierAugment3D(
        freq_mask_prob=0.3,
        phase_noise_std=0.1,
        intensity_scaling_range=(0.8, 1.2),
        prob=1.0
    )
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Apply augmentations
    intensity_aug = intensity_augmenter(volume_tensor).numpy()
    
    torch.manual_seed(42)
    np.random.seed(42)
    spatial_aug = spatial_augmenter(volume_tensor).numpy()
    
    torch.manual_seed(42)
    np.random.seed(42)
    combined_aug = combined_augmenter(volume_tensor).numpy()
    
    torch.manual_seed(42)
    np.random.seed(42)
    fourier_aug = fourier_augmenter(volume_tensor).numpy()
    
    # Visualize comparison
    titles = ['Original', 'Intensity Only', 'Spatial Only', 'Combined', 'Fourier']
    visualize_volumes([volume, intensity_aug, spatial_aug, combined_aug, fourier_aug], titles,
                      save_path='augmentation_output/augmentation_comparison.png')


def main():
    """Main function to demonstrate MONAI-based augmentations."""
    # Create output directory
    os.makedirs('augmentation_output', exist_ok=True)
    
    # Generate a synthetic volume
    volume = generate_synthetic_volume(size=64, num_spheres=8, channels=1)
    print(f"Generated volume with shape {volume.shape}")
    
    # Demonstrate augmentation composer
    demonstrate_augmentation_composer()
    
    # Run and visualize multiple augmentations
    run_multiple_augmentations(volume)
    
    # Compare different augmentation types
    compare_augmentation_types()
    
    print("Visualization images saved to the 'augmentation_output' directory.")


if __name__ == "__main__":
    main()
