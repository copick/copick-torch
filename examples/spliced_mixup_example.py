"""
Example demonstrating the use of SplicedMixupDataset with visualization.

This script shows how to use the SplicedMixupDataset for balanced sampling and synthetic-experimental data splicing.
"""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from copick_torch import SplicedMixupDataset, setup_logging
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import os

def main():
    # Set up logging
    setup_logging()
    
    # Create cache directory if it doesn't exist
    os.makedirs("./cache", exist_ok=True)

    # Basic usage of SplicedMixupDataset
    dataset = SplicedMixupDataset(
        exp_dataset_id=10440,         # Experimental dataset ID
        synth_dataset_id=10441,       # Synthetic dataset ID
        synth_run_id="16487",         # Synthetic run ID
        overlay_root="/tmp/test/",     # Overlay root directory
        boxsize=(48, 48, 48),          # Size of the subvolumes
        augment=True,                 # Enable basic augmentations
        cache_dir='./cache',          # Cache directory
        cache_format='parquet',       # Cache format
        voxel_spacing=10.0,           # Voxel spacing
        include_background=True,      # Include background samples
        background_ratio=0.2,         # Background ratio
        min_background_distance=48,   # Minimum distance from particles for background
        blend_sigma=2.0,              # Sigma for Gaussian blending
        mixup_alpha=0.2               # Alpha parameter for mixup
    )

    # Print dataset information
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.keys()}")
    
    # Show class distribution
    distribution = dataset.get_class_distribution()
    print("\nClass Distribution:")
    for class_name, count in distribution.items():
        print(f"  {class_name}: {count} samples")

    # Use class weights for balanced sampling
    sample_weights = dataset.get_sample_weights()
    
    # Create a weighted sampler for balanced sampling
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )

    # Create data loader with balanced sampling
    batch_size = 8
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4 if multiprocessing.cpu_count() > 4 else 1
    )

    # Get a batch for visualization
    for inputs, labels in dataloader:
        visualize_batch(inputs, labels, dataset.keys())
        break

def visualize_batch(inputs, labels, class_names):
    """Visualize a batch of 3D subvolumes."""
    # Convert class indices to class names
    label_names = []
    for l in labels.numpy():
        if l == -1:
            label_names.append("background")
        elif 0 <= l < len(class_names):
            label_names.append(class_names[l])
        else:
            label_names.append(f"unknown_{l}")
    
    # Create visualization
    batch_size = inputs.shape[0]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(min(batch_size, 8)):
        # Get central slice of each volume
        middle_slice = inputs[i, 0, inputs.shape[2]//2, :, :]
        
        # Display the slice
        axes[i].imshow(middle_slice.numpy(), cmap='gray')
        axes[i].set_title(f"Class: {label_names[i]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("spliced_mixup_visualization.png")
    plt.show()
    
    print(f"Visualization saved to spliced_mixup_visualization.png")

if __name__ == "__main__":
    # This is required for multiprocessing on macOS
    multiprocessing.freeze_support()
    
    # Run the example
    main()
