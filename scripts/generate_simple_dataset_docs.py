#!/usr/bin/env python
"""
Script to generate documentation for the CopickDataset classes.

This script creates both a SimpleCopickDataset and MinimalCopickDataset instance using
the experimental dataset ID 10440, then saves an example volume from each class with the
correct labels to provide a visual reference of the dataset contents.
"""

import logging
import os
import random
from collections import defaultdict
from pathlib import Path

import copick
import matplotlib.pyplot as plt
import numpy as np
import torch

# Import necessary classes
from copick_torch import MinimalCopickDataset, SimpleCopickDataset, setup_logging

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def visualize_volume(volume, title, output_path, cmap="gray"):
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
    axes[0, 0].axis("off")

    axes[0, 1].imshow(volume[:, y_slice, :], cmap=cmap)
    axes[0, 1].set_title(f"XZ Plane (Y={y_slice})")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(volume[:, :, x_slice], cmap=cmap)
    axes[0, 2].set_title(f"YZ Plane (X={x_slice})")
    axes[0, 2].axis("off")

    # Sum projections
    axes[1, 0].imshow(np.sum(volume, axis=0), cmap=cmap)
    axes[1, 0].set_title("XY Sum Projection")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(np.sum(volume, axis=1), cmap=cmap)
    axes[1, 1].set_title("XZ Sum Projection")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(np.sum(volume, axis=2), cmap=cmap)
    axes[1, 2].set_title("YZ Sum Projection")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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

        return object_names
    except Exception as e:
        print(f"Error getting pickable objects: {e}")
        return []


def collect_examples_from_dataset(dataset, include_background=True):
    """
    Collect examples from the dataset for each class.

    Args:
        dataset: Dataset to collect examples from
        include_background: Whether to include background class

    Returns:
        Dictionary mapping class names to example volumes
    """
    print("Collecting examples from dataset...")
    examples = {}
    class_indices = defaultdict(list)

    # Collect indices by class
    for i in range(len(dataset)):
        volume, label = dataset[i]

        # Get class name from label
        if label == -1:
            class_name = "background"
            if not include_background:
                continue
        else:
            try:
                # Make sure we're within bounds
                if label < len(dataset.keys()):
                    class_name = dataset.keys()[label]
                else:
                    print(f"WARNING: Label {label} is out of bounds for dataset keys. Skipping.")
                    continue
            except Exception as e:
                print(f"Error getting class name for label {label}: {e}")
                continue

        # Store the index by class name
        class_indices[class_name].append(i)

    # Get one example from each class
    for class_name, indices in class_indices.items():
        if indices:
            # Choose a random index
            idx = random.choice(indices)
            volume, _ = dataset[idx]
            examples[class_name] = volume
            print(f"  Added example for {class_name}")

    return examples


def main():
    """Main function to generate the documentation."""
    # Set up logging
    setup_logging()

    # Create output directories
    simple_output_dir = Path("docs/simple_dataset_examples")
    minimal_output_dir = Path("docs/minimal_dataset_examples")

    simple_output_dir.mkdir(parents=True, exist_ok=True)
    minimal_output_dir.mkdir(parents=True, exist_ok=True)

    # Create markdown files
    simple_md_file = simple_output_dir / "README.md"
    minimal_md_file = minimal_output_dir / "README.md"

    # Create cache directory if it doesn't exist
    os.makedirs("./cache", exist_ok=True)

    # Get pickable objects directly from the dataset first
    dataset_id = 10440  # Experimental dataset ID
    overlay_root = "/tmp/test/"
    pickable_objects = get_pickable_objects_from_dataset(dataset_id, overlay_root)  # noqa: F841

    # Create SimpleCopickDataset
    print("\nLoading SimpleCopickDataset...")
    simple_dataset = SimpleCopickDataset(
        dataset_id=dataset_id,  # Experimental dataset ID
        overlay_root=overlay_root,  # Overlay root directory
        boxsize=(48, 48, 48),  # Size of the subvolumes
        augment=False,  # Disable augmentations for examples
        cache_dir="./cache",  # Cache directory
        cache_format="parquet",  # Cache format
        voxel_spacing=10.012,  # Voxel spacing (use the exact spacing for best results)
        include_background=True,  # Include background samples
        background_ratio=0.2,  # Background ratio
        min_background_distance=48,  # Minimum distance from particles for background
        max_samples=200,  # Maximum number of samples to generate
    )

    # Print SimpleCopickDataset information
    print(f"SimpleCopickDataset size: {len(simple_dataset)}")
    print(f"SimpleCopickDataset classes: {simple_dataset.keys()}")

    # Show class distribution for SimpleCopickDataset
    simple_distribution = simple_dataset.get_class_distribution()
    print("\nSimpleCopickDataset Class Distribution:")
    for class_name, count in simple_distribution.items():
        print(f"  {class_name}: {count} samples")

    # Create MinimalCopickDataset
    print("\nLoading MinimalCopickDataset...")
    minimal_dataset = MinimalCopickDataset(
        dataset_id=dataset_id,  # Experimental dataset ID
        overlay_root=overlay_root,  # Overlay root directory
        boxsize=(48, 48, 48),  # Size of the subvolumes
        voxel_spacing=10.012,  # Voxel spacing
        include_background=True,  # Include background samples
        background_ratio=0.2,  # Background ratio
        min_background_distance=48,  # Minimum distance from particles for background
        max_samples=200,  # Maximum number of samples to generate
    )

    # Print MinimalCopickDataset information
    print(f"MinimalCopickDataset size: {len(minimal_dataset)}")
    print(f"MinimalCopickDataset classes: {minimal_dataset.keys()}")

    # Show class distribution for MinimalCopickDataset
    minimal_distribution = minimal_dataset.get_class_distribution()
    print("\nMinimalCopickDataset Class Distribution:")
    for class_name, count in minimal_distribution.items():
        print(f"  {class_name}: {count} samples")

    # Print comparison of class keys
    print("\nComparison of class keys:")
    simple_keys = set(simple_dataset.keys())
    minimal_keys = set(minimal_dataset.keys())

    print(f"  SimpleCopickDataset: {sorted(simple_keys)}")
    print(f"  MinimalCopickDataset: {sorted(minimal_keys)}")

    if simple_keys == minimal_keys:
        print("  ✓ Class keys are identical")
    else:
        print("  ✗ Class keys differ")
        print(f"  Only in SimpleCopickDataset: {simple_keys - minimal_keys}")
        print(f"  Only in MinimalCopickDataset: {minimal_keys - simple_keys}")

    # Collect examples from both datasets
    simple_examples = collect_examples_from_dataset(simple_dataset)
    minimal_examples = collect_examples_from_dataset(minimal_dataset)

    # Compare available classes
    print("\nComparison of available examples:")
    print(f"  SimpleCopickDataset: {sorted(simple_examples.keys())}")
    print(f"  MinimalCopickDataset: {sorted(minimal_examples.keys())}")

    if set(simple_examples.keys()) == set(minimal_examples.keys()):
        print("  ✓ Available examples are identical")
    else:
        print("  ✗ Available examples differ")
        print(f"  Only in SimpleCopickDataset: {set(simple_examples.keys()) - set(minimal_examples.keys())}")
        print(f"  Only in MinimalCopickDataset: {set(minimal_examples.keys()) - set(simple_examples.keys())}")

    # Generate examples for SimpleCopickDataset
    print("\nGenerating examples for SimpleCopickDataset...")
    with open(simple_md_file, "w") as f:
        f.write("# SimpleCopickDataset Examples\n\n")
        f.write(
            "This document shows examples of volumes from each class in the dataset used by the `SimpleCopickDataset` class.\n\n",
        )
        f.write(
            "The visualizations show both central slices (top row) and sum projections (bottom row) in XY, XZ, and YZ planes.\n\n",
        )

        # Process each class
        for class_name, volume in simple_examples.items():
            print(f"Processing SimpleCopickDataset class: {class_name}")

            # Create filename - ensure it exactly matches the class name
            filename = f"{class_name.lower().replace(' ', '_').replace('-', '_')}.png"
            filepath = simple_output_dir / filename

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
        f.write(
            "The dataset used in this example is created using the `SimpleCopickDataset` class with the following parameters:\n\n",
        )
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
        for class_name, count in simple_distribution.items():
            f.write(f"| {class_name} | {count} |\n")

    # Generate examples for MinimalCopickDataset
    print("\nGenerating examples for MinimalCopickDataset...")
    with open(minimal_md_file, "w") as f:
        f.write("# MinimalCopickDataset Examples\n\n")
        f.write(
            "This document shows examples of volumes from each class in the dataset used by the `MinimalCopickDataset` class.\n\n",
        )
        f.write(
            "The visualizations show both central slices (top row) and sum projections (bottom row) in XY, XZ, and YZ planes.\n\n",
        )

        # Process each class
        for class_name, volume in minimal_examples.items():
            print(f"Processing MinimalCopickDataset class: {class_name}")

            # Create filename - ensure it exactly matches the class name
            filename = f"{class_name.lower().replace(' ', '_').replace('-', '_')}.png"
            filepath = minimal_output_dir / filename

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
        f.write(
            "The dataset used in this example is created using the `MinimalCopickDataset` class with the following parameters:\n\n",
        )
        f.write("```python\n")
        f.write("dataset = MinimalCopickDataset(\n")
        f.write("    dataset_id=10440,          # Experimental dataset ID\n")
        f.write("    overlay_root='/tmp/test/', # Overlay root directory\n")
        f.write("    boxsize=(48, 48, 48),      # Size of the subvolumes\n")
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
        for class_name, count in minimal_distribution.items():
            f.write(f"| {class_name} | {count} |\n")

        # Add comparison section
        f.write("\n## Comparison with SimpleCopickDataset\n\n")
        f.write("The `MinimalCopickDataset` differs from `SimpleCopickDataset` in the following ways:\n\n")
        f.write("1. **No caching**: Data is loaded on-the-fly instead of being cached to disk\n")
        f.write("2. **No augmentation**: No data augmentation is applied\n")
        f.write("3. **Simplified implementation**: Minimal dependencies and focused on correct label mapping\n")
        f.write("4. **Direct subvolume extraction**: Subvolumes are extracted directly from the zarr array\n\n")

        f.write("### Class Key Comparison\n\n")
        f.write("| Dataset | Class Keys |\n")
        f.write("|---------|------------|\n")
        f.write(f"| SimpleCopickDataset | {', '.join(sorted(simple_keys))} |\n")
        f.write(f"| MinimalCopickDataset | {', '.join(sorted(minimal_keys))} |\n\n")

        if simple_keys == minimal_keys:
            f.write("✓ Class keys are identical between both datasets.\n\n")
        else:
            f.write("✗ Class keys differ between datasets.\n\n")
            f.write(f"Only in SimpleCopickDataset: {', '.join(simple_keys - minimal_keys)}\n\n")
            f.write(f"Only in MinimalCopickDataset: {', '.join(minimal_keys - simple_keys)}\n\n")

    print("\nDocumentation generated successfully:")
    print(f"  - SimpleCopickDataset examples: {simple_md_file}")
    print(f"  - MinimalCopickDataset examples: {minimal_md_file}")


if __name__ == "__main__":
    main()
