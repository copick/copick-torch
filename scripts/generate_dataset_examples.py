#!/usr/bin/env python
"""
Script to generate documentation examples for the MinimalCopickDataset class.

This script creates a MinimalCopickDataset instance with limited examples for each class
using the experimental dataset ID 10440, then saves an example volume from each class
to provide a visual reference of the dataset contents.
"""

import logging
import os
import random
import time
from pathlib import Path

import copick
import matplotlib.pyplot as plt
import numpy as np
import torch

# Import necessary classes
from copick_torch import MinimalCopickDataset, setup_logging

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


def collect_examples_from_dataset(dataset, include_background=True, max_examples=1):
    """
    Collect examples from the dataset for each class.

    Args:
        dataset: Dataset to collect examples from
        include_background: Whether to include background class
        max_examples: Maximum number of examples to collect per class (for efficiency)

    Returns:
        Dictionary mapping class names to example volumes
    """
    print("Collecting examples from dataset...")
    examples = {}
    class_indices = {class_name: [] for class_name in dataset.keys()}

    # Initialize counters for each class
    class_counts = dict.fromkeys(dataset.keys(), 0)
    max_samples_reached = False

    # Collect indices by class
    print("Scanning dataset for examples...")
    for i in range(len(dataset)):
        # Check if we've collected enough examples for each class
        if max_samples_reached:
            break

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

        # If we already have enough examples for this class, skip
        if class_counts.get(class_name, 0) >= max_examples:
            continue

        # Store the index by class name
        class_indices[class_name].append(i)
        class_counts[class_name] += 1

        # Check if we've collected enough examples for all classes
        max_samples_reached = all(count >= max_examples for count in class_counts.values())
        if max_samples_reached:
            print(f"Collected {max_examples} examples for each class. Stopping scan.")

    # Get examples from the indices
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
    start_time = time.time()

    # Set up logging
    setup_logging()

    # Create output directory
    output_dir = Path("docs/dataset_examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create markdown file
    md_file = output_dir / "README.md"

    # Get pickable objects directly from the dataset first
    dataset_id = 10440  # Experimental dataset ID
    overlay_root = "/tmp/test/"
    pickable_objects = get_pickable_objects_from_dataset(dataset_id, overlay_root)  # noqa: F841

    # Create MinimalCopickDataset with a very small max_samples value for efficiency
    print("\nLoading MinimalCopickDataset...")
    dataset = MinimalCopickDataset(
        dataset_id=dataset_id,  # Experimental dataset ID
        overlay_root=overlay_root,  # Overlay root directory
        boxsize=(48, 48, 48),  # Size of the subvolumes
        voxel_spacing=10.012,  # Voxel spacing
        include_background=True,  # Include background samples
        background_ratio=0.2,  # Background ratio
        min_background_distance=48,  # Minimum distance from particles for background
        max_samples=50,  # Limit samples for faster processing
    )

    # Print dataset information
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.keys()}")

    # Show class distribution
    distribution = dataset.get_class_distribution()
    print("\nClass Distribution:")
    for class_name, count in distribution.items():
        print(f"  {class_name}: {count} samples")

    # Collect a single example for each class for efficiency
    examples = collect_examples_from_dataset(dataset, max_examples=1)

    # Generate examples for the dataset
    print("\nGenerating examples for dataset...")
    with open(md_file, "w") as f:
        f.write("# CopickDataset Examples\n\n")
        f.write(
            "This document shows examples of volumes from each class in the dataset used by the `MinimalCopickDataset` class.\n\n",
        )
        f.write(
            "The visualizations show both central slices (top row) and sum projections (bottom row) in XY, XZ, and YZ planes.\n\n",
        )

        # Process each class
        for class_name, volume in examples.items():
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
        f.write(")\n")
        f.write("```\n\n")

        # Add class distribution information
        f.write("### Class Distribution\n\n")
        f.write("| Class | Count |\n")
        f.write("|-------|-------|\n")
        for class_name, count in distribution.items():
            f.write(f"| {class_name} | {count} |\n")

    end_time = time.time()
    print(f"\nDocumentation generated successfully in {end_time - start_time:.2f} seconds.")
    print(f"Examples saved to {output_dir}")
    print(f"Markdown file saved to {md_file}")


if __name__ == "__main__":
    main()
