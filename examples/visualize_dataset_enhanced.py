#!/usr/bin/env python
"""
Enhanced example that loads a saved MinimalCopickDataset and creates a visual report
with orthogonal views of the central section and sum projections with an elegant layout.

Usage:
    python visualize_dataset_enhanced.py --dataset_dir /path/to/saved/dataset --output_file report.png
"""

import argparse
import logging
import os
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

from copick_torch.minimal_dataset import MinimalCopickDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize a saved MinimalCopickDataset with enhanced layout")

    # Required arguments
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory where the dataset was saved")

    # Optional arguments
    parser.add_argument(
        "--output_file",
        type=str,
        default="dataset_visualization_enhanced.png",
        help="Output file for the visualization (default: dataset_visualization_enhanced.png)",
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=2,
        help="Number of samples to display per class (default: 2)",
    )
    parser.add_argument("--dpi", type=int, default=150, help="DPI for the output image (default: 150)")
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Colormap to use for visualization (default: viridis)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def visualize_sample(fig, outer_grid, row, volume, title=None, cmap="viridis"):
    """
    Visualize a 3D sample with orthogonal views and sum projections in an elegant layout.

    Args:
        fig: Matplotlib figure
        outer_grid: GridSpec for placing the subplots
        row: Row index in the outer grid
        volume: 3D numpy array to visualize
        title: Optional title for the visualization
        cmap: Colormap to use
    """
    # Extract dimensions
    depth, height, width = volume.shape
    center_z = depth // 2
    center_y = height // 2
    center_x = width // 2

    # Create a nested GridSpec for this row
    inner_grid = GridSpec(2, 3, wspace=0.3, hspace=0.3, figure=fig, subplot_spec=outer_grid[row, :])

    # Calculate shared min/max for consistent scaling
    vmin = np.min(volume)
    vmax = np.max(volume)

    # Row 1: Orthogonal central slices
    ax_xy = fig.add_subplot(inner_grid[0, 0])
    im_xy = ax_xy.imshow(volume[center_z, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
    ax_xy.set_title("XY Slice (Center)")
    ax_xy.set_xlabel("X")
    ax_xy.set_ylabel("Y")
    ax_xy.set_xticks([])
    ax_xy.set_yticks([])

    ax_xz = fig.add_subplot(inner_grid[0, 1])
    ax_xz.imshow(volume[:, center_y, :], cmap=cmap, vmin=vmin, vmax=vmax)
    ax_xz.set_title("XZ Slice (Center)")
    ax_xz.set_xlabel("X")
    ax_xz.set_ylabel("Z")
    ax_xz.set_xticks([])
    ax_xz.set_yticks([])

    ax_yz = fig.add_subplot(inner_grid[0, 2])
    ax_yz.imshow(volume[:, :, center_x], cmap=cmap, vmin=vmin, vmax=vmax)
    ax_yz.set_title("YZ Slice (Center)")
    ax_yz.set_xlabel("Y")
    ax_yz.set_ylabel("Z")
    ax_yz.set_xticks([])
    ax_yz.set_yticks([])

    # Row 2: Sum projections
    ax_xy_sum = fig.add_subplot(inner_grid[1, 0])
    ax_xy_sum.imshow(np.sum(volume, axis=0), cmap=cmap)
    ax_xy_sum.set_title("XY Projection (Sum)")
    ax_xy_sum.set_xlabel("X")
    ax_xy_sum.set_ylabel("Y")
    ax_xy_sum.set_xticks([])
    ax_xy_sum.set_yticks([])

    ax_xz_sum = fig.add_subplot(inner_grid[1, 1])
    ax_xz_sum.imshow(np.sum(volume, axis=1), cmap=cmap)
    ax_xz_sum.set_title("XZ Projection (Sum)")
    ax_xz_sum.set_xlabel("X")
    ax_xz_sum.set_ylabel("Z")
    ax_xz_sum.set_xticks([])
    ax_xz_sum.set_yticks([])

    ax_yz_sum = fig.add_subplot(inner_grid[1, 2])
    ax_yz_sum.imshow(np.sum(volume, axis=2), cmap=cmap)
    ax_yz_sum.set_title("YZ Projection (Sum)")
    ax_yz_sum.set_xlabel("Y")
    ax_yz_sum.set_ylabel("Z")
    ax_yz_sum.set_xticks([])
    ax_yz_sum.set_yticks([])

    # Add a color bar
    cbar_ax = fig.add_axes(
        [0.92, inner_grid.get_position(fig).y0 + inner_grid.get_height() * 0.1, 0.01, inner_grid.get_height() * 0.8],
    )
    fig.colorbar(im_xy, cax=cbar_ax)

    # Add a label for the class
    if title:
        label_ax = fig.add_subplot(inner_grid[:, :])
        label_ax.axis("off")
        label_ax.text(
            -0.1,
            0.5,
            title,
            rotation=90,
            ha="center",
            va="center",
            transform=label_ax.transAxes,
            fontsize=12,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.5"),
        )


def main():
    """Main function to visualize the dataset with enhanced layout."""
    # Parse command line arguments
    args = parse_args()

    # Set log level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Log the parameters
    logger.info("Visualizing dataset with the following parameters:")
    logger.info(f"  Dataset directory: {args.dataset_dir}")
    logger.info(f"  Output file: {args.output_file}")
    logger.info(f"  Samples per class: {args.samples_per_class}")
    logger.info(f"  Colormap: {args.cmap}")

    try:
        # Load the dataset
        logger.info(f"Loading dataset from {args.dataset_dir}...")
        dataset = MinimalCopickDataset.load(args.dataset_dir)

        # Get class distribution
        distribution = dataset.get_class_distribution()
        logger.info("Class distribution:")
        for class_name, count in distribution.items():
            logger.info(f"  {class_name}: {count} samples")

        # Group samples by class
        samples_by_class = defaultdict(list)

        # Iterate through the dataset
        logger.info("Grouping samples by class...")
        for i in range(len(dataset)):
            # Get the sample and label
            sample, label = dataset[i]

            # Convert to numpy and remove channel dimension
            sample_np = sample.squeeze(0).numpy()

            # Find the class name for this label
            class_name = "background" if label == -1 else None
            if class_name is None:
                for name, idx in dataset._name_to_label.items():
                    if idx == label:
                        class_name = name
                        break

            # Skip if no class name found
            if class_name is None:
                logger.warning(f"No class name found for label {label}, skipping")
                continue

            # Add to samples by class
            samples_by_class[class_name].append((i, sample_np))

        # Set up aesthetics for the plot
        plt.style.use("default")
        mpl.rcParams["axes.grid"] = False
        mpl.rcParams["figure.facecolor"] = "white"
        mpl.rcParams["axes.facecolor"] = "white"

        # Determine the number of classes and samples
        num_classes = len(samples_by_class)
        total_samples = sum(min(len(samples), args.samples_per_class) for samples in samples_by_class.values())

        logger.info(f"Creating visualization with {total_samples} samples from {num_classes} classes...")

        # Create a figure for visualization
        fig = plt.figure(figsize=(12, 5 * total_samples), constrained_layout=True)
        outer_grid = GridSpec(total_samples, 1, figure=fig, hspace=0.4)

        # Add title
        dataset_name = os.path.basename(args.dataset_dir)
        fig.suptitle(f"Dataset Visualization: {dataset_name}", fontsize=16, y=0.98)

        # Add dataset info as text
        info_text = f"Total samples: {len(dataset)}\n"
        info_text += "Class distribution:\n"
        info_text += "\n".join([f"  • {name}: {count}" for name, count in distribution.items()])
        fig.text(
            0.02,
            0.96,
            info_text,
            fontsize=10,
            va="top",
            ha="left",
            bbox=dict(facecolor="whitesmoke", alpha=0.8, boxstyle="round,pad=0.5"),
        )

        # Plot samples for each class
        row = 0
        for class_name, samples in samples_by_class.items():
            # Get up to samples_per_class samples
            num_samples = min(len(samples), args.samples_per_class)

            # If there are no samples, skip this class
            if num_samples == 0:
                logger.warning(f"No samples found for class {class_name}, skipping")
                continue

            # Select random samples
            indices = np.random.choice(len(samples), num_samples, replace=False)
            selected_samples = [samples[i] for i in indices]

            # Visualize each sample
            for i, (sample_idx, sample) in enumerate(selected_samples):
                # Add sample title with index
                title = f"{class_name} ({sample_idx})"

                # Visualize the sample
                visualize_sample(fig, outer_grid, row, sample, title=title, cmap=args.cmap)
                row += 1

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])

        # Save the figure
        logger.info(f"Saving visualization to {args.output_file}...")
        plt.savefig(args.output_file, dpi=args.dpi, bbox_inches="tight")
        logger.info("Visualization saved successfully")

    except Exception as e:
        logger.exception(f"Error visualizing dataset: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
