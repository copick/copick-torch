#!/usr/bin/env python
"""
Script to generate documentation for the spliced_mixup_example.py dataset.

This script extracts examples from each class in the dataset and applies
various augmentations to demonstrate their effects. It saves visualizations
showing both central slices and sum projections in orthogonal views.
"""

import os
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, WeightedRandomSampler

# Import necessary classes
from copick_torch import SplicedMixupDataset, setup_logging
from copick_torch.augmentations import FourierAugment3D

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# Define augmentation functions
def no_augmentation(volume):
    """Return the volume with no augmentation."""
    return volume.clone()


def brightness_adjustment(volume, delta=0.5):
    """Apply brightness adjustment."""
    return volume + delta


def gaussian_blur(volume, sigma=1.0):
    """Apply Gaussian blur."""
    # Convert to numpy for scipy.ndimage
    is_tensor = isinstance(volume, torch.Tensor)
    if is_tensor:
        volume_np = volume.squeeze(0).numpy()
    else:
        volume_np = volume

    # Apply Gaussian blur
    blurred = gaussian_filter(volume_np, sigma=sigma)

    # Convert back to tensor if needed
    if is_tensor:
        return torch.from_numpy(blurred).unsqueeze(0)
    return blurred


def intensity_scaling(volume, factor=1.5):
    """Apply intensity scaling."""
    return volume * factor


def flip_augmentation(volume, axis=0):
    """Apply flip along specified axis."""
    if isinstance(volume, torch.Tensor):
        return torch.flip(volume, dims=[axis + 1])  # +1 for channel dimension
    return np.flip(volume, axis=axis)


def rotation_augmentation(volume, k=1, axes=(0, 1)):
    """Apply rotation augmentation."""
    if isinstance(volume, torch.Tensor):
        # For tensors, transpose axes and perform rotation
        volume = volume.squeeze(0)
        if axes == (0, 1):
            volume = torch.rot90(volume, k=k, dims=[0, 1])
        elif axes == (0, 2):
            volume = torch.rot90(volume, k=k, dims=[0, 2])
        elif axes == (1, 2):
            volume = torch.rot90(volume, k=k, dims=[1, 2])
        return volume.unsqueeze(0)
    else:
        # For numpy arrays
        return np.rot90(volume, k=k, axes=axes)


def fourier_augmentation(volume):
    """Apply Fourier domain augmentation using MONAI-based implementation."""
    # Create the augmentation object
    fourier_aug = FourierAugment3D(
        freq_mask_prob=0.3,
        phase_noise_std=0.1,
        intensity_scaling_range=(0.8, 1.2),
        prob=1.0,  # Always apply the transform for demonstration
    )

    # Apply the augmentation
    if isinstance(volume, torch.Tensor):
        # MONAI transform expects tensor input
        augmented = fourier_aug(volume)
        return augmented
    else:
        # Convert numpy to tensor, apply transform, then convert back
        tensor_vol = torch.from_numpy(volume)
        if len(tensor_vol.shape) == 3:  # Add channel dimension if needed
            tensor_vol = tensor_vol.unsqueeze(0)
        augmented = fourier_aug(tensor_vol)
        if len(augmented.shape) == 4:  # Remove channel dimension if added
            augmented = augmented.squeeze(0)
        return augmented.numpy()


# Define the augmentations with their names
AUGMENTATIONS = [
    ("Original", no_augmentation),
    ("Brightness (+0.5)", lambda v: brightness_adjustment(v, delta=0.5)),
    ("Brightness (-0.5)", lambda v: brightness_adjustment(v, delta=-0.5)),
    ("Gaussian Blur (σ=1.0)", lambda v: gaussian_blur(v, sigma=1.0)),
    ("Intensity Scaling (1.5x)", lambda v: intensity_scaling(v, factor=1.5)),
    ("Intensity Scaling (0.5x)", lambda v: intensity_scaling(v, factor=0.5)),
    ("Flip (Z axis)", lambda v: flip_augmentation(v, axis=0)),
    ("Rotation (90°, XY plane)", lambda v: rotation_augmentation(v, k=1, axes=(1, 2))),
    ("MONAI Fourier Augmentation", fourier_augmentation),
]


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


def main():
    """Main function to generate the documentation."""
    # Set up logging
    setup_logging()

    # Create output directory
    output_dir = Path("docs/augmentation_examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create markdown file
    md_file = output_dir / "README.md"

    # Create cache directory if it doesn't exist
    os.makedirs("./cache", exist_ok=True)

    print("Loading dataset...")
    # Create SplicedMixupDataset
    dataset = SplicedMixupDataset(
        exp_dataset_id=10440,  # Experimental dataset ID
        synth_dataset_id=10441,  # Synthetic dataset ID
        synth_run_id="16487",  # Synthetic run ID
        overlay_root="/tmp/test/",  # Overlay root directory
        boxsize=(48, 48, 48),  # Size of the subvolumes
        augment=False,  # Disable basic augmentations for examples
        cache_dir="./cache",  # Cache directory
        cache_format="parquet",  # Cache format
        voxel_spacing=10.012,  # Voxel spacing (use the exact spacing for best results)
        include_background=True,  # Include background samples
        background_ratio=0.2,  # Background ratio
        min_background_distance=48,  # Minimum distance from particles for background
        blend_sigma=2.0,  # Controls the standard deviation of Gaussian blending at boundaries
        mixup_alpha=0.2,  # Alpha parameter for mixup
        max_samples=100,  # Maximum number of samples to generate
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
        _, label = dataset[i]
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
    with open(md_file, "w") as f:
        f.write("# Copick-Torch Augmentation Examples\n\n")
        f.write(
            "This document shows examples of various augmentations applied to the dataset used in the `spliced_mixup_example.py` example.\n\n",
        )
        f.write("For each class, we show the original volume and various augmentations applied to it.\n")
        f.write(
            "The visualizations show both central slices (top row) and sum projections (bottom row) in XY, XZ, and YZ planes.\n\n",
        )

        # Process each class
        for class_name, volume in class_examples.items():
            print(f"Processing class: {class_name}")
            f.write(f"## Class: {class_name}\n\n")

            # Apply each augmentation
            for aug_name, aug_func in AUGMENTATIONS:
                print(f"  Applying augmentation: {aug_name}")
                try:
                    # Apply augmentation
                    aug_volume = aug_func(volume)

                    # Create filename
                    filename = f"{class_name}_{aug_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace(',', '').replace('°', '')}.png"
                    filepath = output_dir / filename

                    # Normalize for visualization
                    if isinstance(aug_volume, torch.Tensor):
                        vis_volume = aug_volume.squeeze(0)
                        if torch.std(vis_volume) > 0:
                            vis_volume = (vis_volume - torch.mean(vis_volume)) / torch.std(vis_volume)
                        vis_volume = vis_volume.numpy()
                    else:
                        if np.std(aug_volume) > 0:
                            vis_volume = (aug_volume - np.mean(aug_volume)) / np.std(aug_volume)
                        else:
                            vis_volume = aug_volume

                    # Visualize and save
                    visualize_volume(vis_volume, f"{class_name} - {aug_name}", filepath)

                    # Add to markdown
                    f.write(f"### {aug_name}\n\n")
                    f.write(f"![{class_name} - {aug_name}](./{filename})\n\n")

                except Exception as e:
                    print(f"Error applying {aug_name} to {class_name}: {str(e)}")

        # Add information about the dataset
        f.write("## Dataset Information\n\n")
        f.write(
            "The dataset used in this example is created using the `SplicedMixupDataset` class with the following parameters:\n\n",
        )
        f.write("```python\n")
        f.write("dataset = SplicedMixupDataset(\n")
        f.write("    exp_dataset_id=10440,         # Experimental dataset ID\n")
        f.write("    synth_dataset_id=10441,       # Synthetic dataset ID\n")
        f.write('    synth_run_id="16487",         # Synthetic run ID\n')
        f.write('    overlay_root="/tmp/test/",    # Overlay root directory\n')
        f.write("    boxsize=(48, 48, 48),         # Size of the subvolumes\n")
        f.write("    augment=True,                 # Enable basic augmentations\n")
        f.write("    cache_dir='./cache',          # Cache directory\n")
        f.write("    cache_format='parquet',       # Cache format\n")
        f.write("    voxel_spacing=10.012,         # Voxel spacing\n")
        f.write("    include_background=True,      # Include background samples\n")
        f.write("    background_ratio=0.2,         # Background ratio\n")
        f.write("    min_background_distance=48,   # Minimum distance from particles for background\n")
        f.write("    blend_sigma=2.0,              # Controls Gaussian blending at boundaries\n")
        f.write("    mixup_alpha=0.2,              # Alpha parameter for mixup\n")
        f.write("    max_samples=100               # Maximum number of samples to generate\n")
        f.write(")\n")
        f.write("```\n\n")

        # Add class distribution information
        f.write("### Class Distribution\n\n")
        f.write("| Class | Count |\n")
        f.write("|-------|-------|\n")
        for class_name, count in distribution.items():
            f.write(f"| {class_name} | {count} |\n")

    print(f"Documentation generated successfully. Markdown file saved to {md_file}")


if __name__ == "__main__":
    main()
