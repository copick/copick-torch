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
        voxel_spacing=10.012,           # Voxel spacing (use the exact spacing for best results)
        include_background=True,      # Include background samples
        background_ratio=0.2,         # Background ratio
        min_background_distance=48,   # Minimum distance from particles for background
        blend_sigma=2.0,              # Sigma for Gaussian blending
        mixup_alpha=0.2,              # Alpha parameter for mixup
        max_samples=100               # Maximum number of samples to generate
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
        num_workers=0  # Use single process data loading to avoid pickling issues
    )

    # Try to load a single batch for visualization
    try:
        batch = next(iter(dataloader))
        inputs, labels = batch
        visualize_batch(inputs, labels, dataset.keys())
    except Exception as e:
        print(f"\nError during visualization: {str(e)}")
        # Try to extract a single item directly from the dataset
        print("\nTrying to extract a single item directly from the dataset...")
        try:
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"Successfully extracted single item with shape {sample[0].shape}")
                # Create a mini-batch of 1 item for visualization
                inputs = sample[0].unsqueeze(0)  # Add batch dimension
                labels = torch.tensor([sample[1]])  # Convert to tensor with batch dimension
                visualize_batch(inputs, labels, dataset.keys())
            else:
                print("Dataset is empty.")
        except Exception as e2:
            print(f"Error extracting single item: {str(e2)}")

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
    fig, axes = plt.subplots(3, 4, figsize=(16, 12)) # 3 rows for XY, YZ, XZ planes
    
    for i in range(min(batch_size, 4)):
        # Get central slices along each axis
        voldata = inputs[i, 0].numpy()
        slice_z = voldata.shape[0] // 2
        slice_y = voldata.shape[1] // 2
        slice_x = voldata.shape[2] // 2
        
        # Display XY plane (Z slice)
        xy_slice = voldata[slice_z, :, :]
        axes[0, i].imshow(xy_slice, cmap='gray')
        axes[0, i].set_title(f"Class: {label_names[i]}\nXY Plane (Z={slice_z})")
        axes[0, i].axis('off')
        
        # Display YZ plane (X slice)
        yz_slice = voldata[:, :, slice_x]
        axes[1, i].imshow(yz_slice, cmap='gray')
        axes[1, i].set_title(f"YZ Plane (X={slice_x})")
        axes[1, i].axis('off')
        
        # Display XZ plane (Y slice)
        xz_slice = voldata[:, slice_y, :]
        axes[2, i].imshow(xz_slice, cmap='gray')
        axes[2, i].set_title(f"XZ Plane (Y={slice_y})")
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig("spliced_mixup_visualization.png")
    plt.show()
    
    print(f"Visualization saved to spliced_mixup_visualization.png")

if __name__ == "__main__":
    # This is required for multiprocessing on macOS
    multiprocessing.freeze_support()
    
    # Run the example
    main()
