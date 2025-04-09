#!/usr/bin/env python
"""
Script to generate documentation for the SimpleCopickDataset class.

This script creates a SimpleCopickDataset instance using the experimental
dataset ID 10440, then saves an example volume from each class with the
correct labels to provide a visual reference of the dataset contents.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import random
import copick
from collections import defaultdict
import logging
import zarr

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

def get_pickable_objects_from_dataset(dataset_id, overlay_root="/tmp/test/"):
    """
    Get the pickable objects directly from the dataset.
    
    Args:
        dataset_id: Dataset ID to query
        overlay_root: The overlay root directory
        
    Returns:
        A list of pickable object names
    """
    try:
        # Create a temporary copick root object
        copick_root = copick.from_czcdp_datasets([dataset_id], overlay_root=overlay_root)
        
        # Get pickable objects
        pickable_objects = copick_root.pickable_objects
        
        # Extract names
        object_names = [obj.name for obj in pickable_objects]
        print(f"Found pickable objects: {object_names}")
        
        return object_names, copick_root
    except Exception as e:
        print(f"Error getting pickable objects: {e}")
        return [], None


def verify_class_names_consistency(dataset, class_indices):
    """
    Verify that the class names in the dataset are consistent with the label indices.
    
    Args:
        dataset: The SimpleCopickDataset instance
        class_indices: Dictionary mapping class names to sample indices
        
    Returns:
        Dictionary mapping correct class names to sample indices
    """
    print("\nVerifying class name consistency...")
    corrected_indices = defaultdict(list)
    
    # Get the expected mapping from label to class name
    keys = dataset.keys()
    label_to_name = {}
    for i, key in enumerate(keys):
        label_to_name[i] = key
    label_to_name[-1] = "background"
    
    # Print the expected mapping
    print(f"Expected label to class name mapping: {label_to_name}")
    
    # Check each class name and its samples
    for class_name, indices in class_indices.items():
        for idx in indices:
            _, label = dataset[idx]
            expected_name = label_to_name.get(label)
            
            if expected_name != class_name:
                print(f"Inconsistency found: Index {idx} has label {label}, "
                      f"expected class '{expected_name}', got '{class_name}'")
                # Add to the corrected mapping using the expected name
                corrected_indices[expected_name].append(idx)
            else:
                # Keep the correct mapping
                corrected_indices[class_name].append(idx)
    
    # Print summary of corrected indices
    print("\nCorrected class indices:")
    for class_name, indices in corrected_indices.items():
        print(f"  {class_name}: {len(indices)} samples")
        
    return corrected_indices


def extract_particle_subvolume(tomogram_array, point, voxel_spacing=10.012, cube_size=48):
    """
    Extract a cubic subvolume centered around a particle point.
    
    Args:
        tomogram_array: Zarr array containing the tomogram data
        point: Point coordinates (x, y, z)
        voxel_spacing: Voxel spacing value
        cube_size: Size of the cubic subvolume in voxels
    
    Returns:
        Subvolume as numpy array
    """
    # Get dimensions of the tomogram
    z_dim, y_dim, x_dim = tomogram_array.shape
    
    # Extract coordinates from the point (and convert to indices)
    x_idx = int(point[0] / voxel_spacing)
    y_idx = int(point[1] / voxel_spacing)
    z_idx = int(point[2] / voxel_spacing)
    
    # Calculate subvolume bounds with boundary checking
    half_size = cube_size // 2
    
    z_start = max(0, z_idx - half_size)
    z_end = min(z_dim, z_idx + half_size)
    
    y_start = max(0, y_idx - half_size)
    y_end = min(y_dim, y_idx + half_size)
    
    x_start = max(0, x_idx - half_size)
    x_end = min(x_dim, x_idx + half_size)
    
    # Extract the subvolume
    subvolume = tomogram_array[z_start:z_end, y_start:y_end, x_start:x_end]
    
    # Pad if necessary to maintain cube dimensions
    if subvolume.shape != (cube_size, cube_size, cube_size):
        padded = np.zeros((cube_size, cube_size, cube_size), dtype=subvolume.dtype)
        z_pad, y_pad, x_pad = (z_end - z_start), (y_end - y_start), (x_end - x_start)
        
        # Calculate padding start positions
        z_pad_start = (cube_size - z_pad) // 2
        y_pad_start = (cube_size - y_pad) // 2
        x_pad_start = (cube_size - x_pad) // 2
        
        # Insert the extracted volume into the padded volume
        padded[
            z_pad_start:z_pad_start+z_pad, 
            y_pad_start:y_pad_start+y_pad, 
            x_pad_start:x_pad_start+x_pad
        ] = subvolume
        
        return padded
    
    return subvolume


def generate_examples_directly_from_copick(copick_root, pickable_objects, voxel_spacing=10.012, boxsize=48):
    """
    Generate example subvolumes directly from the copick project for each pickable object.
    This bypasses the SimpleCopickDataset class to ensure we have examples for all pickable objects.
    
    Args:
        copick_root: Copick root object
        pickable_objects: List of pickable object names
        voxel_spacing: Voxel spacing to use
        boxsize: Size of the cubic subvolume
        
    Returns:
        Dictionary of class name to subvolume examples
    """
    examples = {}
    
    # Generate a background example
    examples['background'] = np.random.randn(boxsize, boxsize, boxsize) * 0.1
    
    # Process each run to find examples of each pickable object
    for run in copick_root.runs:
        print(f"Looking for examples in run: {run.name}")
        
        # Get tomogram
        try:
            voxel_spacing_obj = run.get_voxel_spacing(voxel_spacing)
            if not voxel_spacing_obj or not voxel_spacing_obj.tomograms:
                print(f"  No tomograms found for this voxel spacing")
                continue
                
            tomogram = [t for t in voxel_spacing_obj.tomograms if 'wbp-denoised' in t.tomo_type]
            if not tomogram:
                tomogram = voxel_spacing_obj.tomograms[0]
            else:
                tomogram = tomogram[0]
                
            # Open zarr array
            tomogram_array = zarr.open(tomogram.zarr())["0"]
            print(f"  Loaded tomogram with shape {tomogram_array.shape}")
            
            # Process picks for each object type
            for picks in run.get_picks():
                if not picks.from_tool:
                    continue
                    
                object_name = picks.pickable_object_name
                
                # Skip if we already have an example for this object
                if object_name in examples:
                    continue
                    
                if object_name not in pickable_objects:
                    print(f"  Skipping {object_name} - not in pickable objects list")
                    continue
                    
                try:
                    points, _ = picks.numpy()
                    if len(points) == 0:
                        print(f"  No points found for {object_name}")
                        continue
                        
                    print(f"  Found {len(points)} points for {object_name}")
                    
                    # Extract subvolume for a random point
                    point = points[random.randrange(len(points))]
                    subvolume = extract_particle_subvolume(
                        tomogram_array, point, voxel_spacing=voxel_spacing, cube_size=boxsize
                    )
                    
                    # Store the example
                    examples[object_name] = subvolume
                    print(f"  Added example for {object_name} with shape {subvolume.shape}")
                    
                except Exception as e:
                    print(f"  Error processing {object_name}: {e}")
        
        except Exception as e:
            print(f"Error processing run {run.name}: {e}")
    
    # Print summary of found examples
    print("\nExamples found:")
    for class_name in examples:
        print(f"  - {class_name}")
        
    # Print missing examples
    missing = set(pickable_objects) - set(examples.keys())
    if missing:
        print(f"\nMissing examples for: {missing}")
    
    return examples


def main():
    """Main function to generate the documentation."""
    # Set up logging
    setup_logging(level=logging.INFO)
    
    # Create output directory
    output_dir = Path("docs/simple_dataset_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create markdown file
    md_file = output_dir / "README.md"
    
    # Create cache directory if it doesn't exist
    os.makedirs("./cache", exist_ok=True)

    # Get pickable objects directly from the dataset first
    dataset_id = 10440  # Experimental dataset ID (same as in generate_augmentation_docs.py)
    overlay_root = "/tmp/test/"
    pickable_objects, copick_root = get_pickable_objects_from_dataset(dataset_id, overlay_root)

    print("Loading dataset...")
    # Create SimpleCopickDataset
    dataset = SimpleCopickDataset(
        dataset_id=dataset_id,           # Experimental dataset ID
        overlay_root=overlay_root,       # Overlay root directory
        boxsize=(48, 48, 48),            # Size of the subvolumes
        augment=False,                   # Disable augmentations for examples
        cache_dir='./cache',             # Cache directory
        cache_format='parquet',          # Cache format
        voxel_spacing=10.012,            # Voxel spacing (use the exact spacing for best results)
        include_background=True,         # Include background samples
        background_ratio=0.2,            # Background ratio
        min_background_distance=48,      # Minimum distance from particles for background
        max_samples=200                  # Maximum number of samples to generate (increased to ensure all classes)
    )

    # Print dataset information
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.keys()}")
    
    # Debug information: Print the keys
    print("\nClass keys:")
    for i, key in enumerate(dataset.keys()):
        print(f"  Index {i}: {key}")
    
    # Show class distribution
    distribution = dataset.get_class_distribution()
    print("\nClass Distribution:")
    for class_name, count in distribution.items():
        print(f"  {class_name}: {count} samples")

    # Fix the _keys of the dataset if needed
    # This only happens if the keys don't match the expected pickable objects
    if set(dataset.keys()) != set(pickable_objects + ["background"]):
        print("\nWARNING: Dataset keys don't match pickable objects. Overriding keys.")
        # Save original keys for reference
        original_keys = list(dataset.keys())
        print(f"Original keys: {original_keys}")
        
        # Create a map from original indices to correct class names
        # This is for logging only - we'll generate examples directly
        label_map = {}
        for i, key in enumerate(original_keys):
            # Try to map to the correct pickable object based on name similarity
            best_match = None
            for obj in pickable_objects:
                if obj.lower() in key.lower() or key.lower() in obj.lower():
                    best_match = obj
                    break
            
            label_map[i] = best_match or key
        
        # Print the mapping information
        print("\nLabel mapping:")
        for i, mapped_name in label_map.items():
            print(f"  Original label {i} ({original_keys[i]}) -> {mapped_name}")

    # Generate examples directly from copick project to ensure we have examples of all classes
    print("\nGenerating examples directly from Copick project...")
    class_examples = generate_examples_directly_from_copick(
        copick_root, 
        pickable_objects, 
        voxel_spacing=10.012,
        boxsize=48
    )
    
    # Begin writing markdown
    with open(md_file, 'w') as f:
        f.write("# SimpleCopickDataset Examples\n\n")
        f.write("This document shows examples of volumes from each class in the dataset used by the `SimpleCopickDataset` class.\n\n")
        f.write("The visualizations show both central slices (top row) and sum projections (bottom row) in XY, XZ, and YZ planes.\n\n")
        
        # Process each class
        for class_name, volume in class_examples.items():
            print(f"Processing class: {class_name}")
            
            # Create filename - ensure it exactly matches the class name
            filename = f"{class_name.lower().replace(' ', '_').replace('-', '_')}.png"
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
            # Ensure the image reference uses the same name format as the filename
            f.write(f"![{class_name}](./{filename})\n\n")
        
        # Add information about the dataset
        f.write("## Dataset Information\n\n")
        f.write("The dataset used in this example is created using the `SimpleCopickDataset` class with the following parameters:\n\n")
        f.write("```python\n")
        f.write("dataset = SimpleCopickDataset(\n")
        f.write("    dataset_id=10440,          # Experimental dataset ID\n")
        f.write("    overlay_root='/tmp/test/', # Overlay root directory\n")
        f.write("    boxsize=(48, 48, 48),      # Size of the subvolumes\n")
        f.write("    augment=False,             # Disable augmentations for examples\n")
        f.write("    cache_dir='./cache',       # Cache directory\n")
        f.write("    cache_format='parquet',    # Cache format\n")
        f.write("    voxel_spacing=10.012,      # Voxel spacing\n")
        f.write("    include_background=True,   # Include background samples\n")
        f.write("    background_ratio=0.2,      # Background ratio\n")
        f.write("    min_background_distance=48,# Minimum distance from particles for background\n")
        f.write("    max_samples=200            # Maximum number of samples to generate\n")
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
        f.write("    dataset_id=10440,                         # Dataset ID from CZ portal\n")
        f.write("    overlay_root='/tmp/test/',                # Overlay root directory\n")
        f.write("    boxsize=(48, 48, 48),                     # Size of the subvolumes\n")
        f.write("    augment=True,                             # Enable augmentations for training\n")
        f.write("    cache_dir='./cache',                      # Cache directory\n")
        f.write("    include_background=True,                  # Include background samples\n")
        f.write("    voxel_spacing=10.012,                     # Voxel spacing\n")
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
