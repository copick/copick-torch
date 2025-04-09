#!/usr/bin/env python
"""
Script to generate documentation for the SimpleCopickDataset class.

This script creates a SimpleCopickDataset instance using the same experimental
dataset as generate_augmentation_docs.py, then saves an example volume
from each class to provide a visual reference of the dataset contents.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import random
from collections import defaultdict

# Import necessary classes
from copick_torch import SimpleCopickDataset, setup_logging

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

def visualize_volume(volume, title, output_path, cmap='gray'):
    """
    Visualize a volume with central slice and sum projection views.
    
    Args:
        volume: 3D volume (torch tensor with shape [C, D, H, W] or numpy array with shape [D, H, W])
        title: Title for the plot
        output_path: Path to save the visualization
        cmap: Colormap to use for visualization
    """
    # Ensure we have a numpy array in the right shape
    if isinstance(volume, torch.Tensor):
        if volume.dim() == 4:  # [C, D, H, W]
            volume = volume.squeeze(0).numpy()
        else:  # [D, H, W]
            volume = volume.numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Get dimensions
    depth, height, width = volume.shape
    
    # Central slices
    z_slice = depth // 2
    y_slice = height // 2
    x_slice = width // 2
    
    # Central slice views
    axes[0, 0].imshow(volume[z_slice, :, :], cmap=cmap)
    axes[0, 0].set_title(f"XY Plane (Z={z_slice})")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(volume[:, y_slice, :], cmap=cmap)
    axes[0, 1].set_title(f"XZ Plane (Y={y_slice})")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(volume[:, :, x_slice], cmap=cmap)
    axes[0, 2].set_title(f"YZ Plane (X={x_slice})")
    axes[0, 2].axis('off')
    
    # Sum projections
    axes[1, 0].imshow(np.sum(volume, axis=0), cmap=cmap)
    axes[1, 0].set_title("XY Sum Projection")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(np.sum(volume, axis=1), cmap=cmap)
    axes[1, 1].set_title("XZ Sum Projection")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(np.sum(volume, axis=2), cmap=cmap)
    axes[1, 2].set_title("YZ Sum Projection")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def main():
    """Main function to generate the documentation."""
    # Set up logging
    setup_logging()
    
    # Create output directory
    output_dir = Path("docs/simple_dataset_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create markdown file
    md_file = output_dir / "README.md"
    
    # Create cache directory if it doesn't exist
    os.makedirs("./cache", exist_ok=True)

    print("Loading dataset...")
    # Create SimpleCopickDataset (use the same dataset ID as in generate_augmentation_docs.py)
    dataset = SimpleCopickDataset(
        dataset_id=10440,           # Experimental dataset ID (same as in generate_augmentation_docs.py)
        overlay_root="/tmp/test/",  # Overlay root directory
        boxsize=(48, 48, 48),       # Size of the subvolumes
        augment=False,              # Disable augmentations for examples
        cache_dir='./cache',        # Cache directory
        cache_format='parquet',     # Cache format
        voxel_spacing=10.012,       # Voxel spacing (use the exact spacing for best results)
        include_background=True,    # Include background samples
        background_ratio=0.2,       # Background ratio
        min_background_distance=48, # Minimum distance from particles for background
        max_samples=100             # Maximum number of samples to generate
    )

    # Print dataset information
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.keys()}")
    
    # Show class distribution
    distribution = dataset.get_class_distribution()
    print("\nClass Distribution:")
    for class_name, count in distribution.items():
        print(f"  {class_name}: {count} samples")

    # Collect one example from each class
    class_examples = {}
    class_indices = defaultdict(list)
    
    # First pass: collect indices by class
    for i in range(len(dataset)):
        volume, label = dataset[i]
        if label == -1:
            class_name = "background"
        else:
            class_name = dataset.keys()[label]
        class_indices[class_name].append(i)
    
    # Second pass: get one example from each class
    for class_name, indices in class_indices.items():
        if indices:
            # Choose a random index from this class
            idx = random.choice(indices)
            volume, _ = dataset[idx]
            class_examples[class_name] = volume
    
    # Begin writing markdown
    with open(md_file, 'w') as f:
        f.write("# SimpleCopickDataset Examples\n\n")
        f.write("This document shows examples of volumes from each class in the dataset used by the `SimpleCopickDataset` class.\n\n")
        f.write("The visualizations show both central slices (top row) and sum projections (bottom row) in XY, XZ, and YZ planes.\n\n")
        
        # Process each class
        for class_name, volume in class_examples.items():
            print(f"Processing class: {class_name}")
            
            # Create filename
            filename = f"{class_name.lower().replace(' ', '_')}.png"
            filepath = output_dir / filename
            
            # Normalize for visualization
            if isinstance(volume, torch.Tensor):
                vis_volume = volume.squeeze(0)
                if torch.std(vis_volume) > 0:
                    vis_volume = (vis_volume - torch.mean(vis_volume)) / torch.std(vis_volume)
                vis_volume = vis_volume.numpy()
            else:
                if np.std(volume) > 0:
                    vis_volume = (volume - np.mean(volume)) / np.std(volume)
                else:
                    vis_volume = volume
            
            # Visualize and save
            visualize_volume(vis_volume, f"Class: {class_name}", filepath)
            
            # Add to markdown
            f.write(f"## Class: {class_name}\n\n")
            f.write(f"![{class_name}](./{filename})\n\n")
        
        # Add information about the dataset
        f.write("## Dataset Information\n\n")
        f.write("The dataset used in this example is created using the `SimpleCopickDataset` class with the following parameters:\n\n")
        f.write("```python\n")
        f.write("dataset = SimpleCopickDataset(\n")
        f.write("    copick_root=None,          # Will be created from the dataset ID\n")
        f.write("    boxsize=(48, 48, 48),      # Size of the subvolumes\n")
        f.write("    augment=False,             # Disable augmentations for examples\n")
        f.write("    cache_dir='./cache',       # Cache directory\n")
        f.write("    cache_format='parquet',    # Cache format\n")
        f.write("    voxel_spacing=10.012,      # Voxel spacing\n")
        f.write("    include_background=True,   # Include background samples\n")
        f.write("    background_ratio=0.2,      # Background ratio\n")
        f.write("    min_background_distance=48,# Minimum distance from particles for background\n")
        f.write("    max_samples=100            # Maximum number of samples to generate\n")
        f.write(")\n")
        f.write("```\n\n")
        
        # Add class distribution information
        f.write("### Class Distribution\n\n")
        f.write("| Class | Count |\n")
        f.write("|-------|-------|\n")
        for class_name, count in distribution.items():
            f.write(f"| {class_name} | {count} |\n")
        
        # Add usage example
        f.write("\n## Usage Example\n\n")
        f.write("Here's an example of how to use the `SimpleCopickDataset` class in a training pipeline:\n\n")
        f.write("```python\n")
        f.write("from copick_torch import SimpleCopickDataset\n")
        f.write("from torch.utils.data import DataLoader, WeightedRandomSampler\n")
        f.write("\n")
        f.write("# Create the dataset\n")
        f.write("dataset = SimpleCopickDataset(\n")
        f.write("    config_path='path/to/copick/config.yaml',  # Path to copick config file\n")
        f.write("    boxsize=(48, 48, 48),                      # Size of the subvolumes\n")
        f.write("    augment=True,                              # Enable augmentations for training\n")
        f.write("    cache_dir='./cache',                       # Cache directory\n")
        f.write("    include_background=True,                   # Include background samples\n")
        f.write("    voxel_spacing=10.0,                        # Voxel spacing\n")
        f.write(")\n")
        f.write("\n")
        f.write("# Print dataset information\n")
        f.write("print(f\"Dataset size: {len(dataset)}\")\n")
        f.write("print(f\"Classes: {dataset.keys()}\")\n")
        f.write("print(f\"Class distribution: {dataset.get_class_distribution()}\")\n")
        f.write("\n")
        f.write("# Create a weighted sampler for balanced training\n")
        f.write("weights = dataset.get_sample_weights()\n")
        f.write("sampler = WeightedRandomSampler(weights, len(weights))\n")
        f.write("\n")
        f.write("# Create a DataLoader\n")
        f.write("dataloader = DataLoader(\n")
        f.write("    dataset,\n")
        f.write("    batch_size=8,\n")
        f.write("    sampler=sampler,\n")
        f.write("    num_workers=4,\n")
        f.write("    pin_memory=True\n")
        f.write(")\n")
        f.write("\n")
        f.write("# Training loop\n")
        f.write("for volume, label in dataloader:\n")
        f.write("    # volume shape: [batch_size, 1, depth, height, width]\n")
        f.write("    # label: [batch_size] class indices\n")
        f.write("    # Your training code here\n")
        f.write("    pass\n")
        f.write("```\n")
    
    print(f"Documentation generated successfully. Markdown file saved to {md_file}")

if __name__ == "__main__":
    main()
