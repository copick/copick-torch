#!/usr/bin/env python
"""
Minimal example that loads a saved MinimalCopickDataset and creates a visual report
with orthogonal views of the central section and sum projections.

Usage:
    python visualize_dataset.py --dataset_dir /path/to/saved/dataset --output_file report.png
"""

import argparse
import logging
import os
from collections import defaultdict

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
    parser = argparse.ArgumentParser(description="Visualize a saved MinimalCopickDataset")

    # Required arguments
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory where the dataset was saved")

    # Optional arguments
    parser.add_argument(
        "--output_file",
        type=str,
        default="dataset_visualization.png",
        help="Output file for the visualization (default: dataset_visualization.png)",
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=2,
        help="Number of samples to display per class (default: 2)",
    )
    parser.add_argument("--dpi", type=int, default=150, help="DPI for the output image (default: 150)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def visualize_sample(ax_list, volume, title=None):
    """
    Visualize a 3D sample with orthogonal views and projections.

    Args:
        ax_list: List of 6 matplotlib axes for plotting
        volume: 3D numpy array to visualize
        title: Optional title for the visualization
    """
    # Extract dimensions
    depth, height, width = volume.shape
    center_z = depth // 2
    center_y = height // 2
    center_x = width // 2

    # Orthogonal central slices
    ax_list[0].imshow(volume[center_z, :, :], cmap="gray")
    ax_list[0].set_title("XY Slice (Central Z)")
    ax_list[0].axis("off")

    ax_list[1].imshow(volume[:, center_y, :], cmap="gray")
    ax_list[1].set_title("XZ Slice (Central Y)")
    ax_list[1].axis("off")

    ax_list[2].imshow(volume[:, :, center_x], cmap="gray")
    ax_list[2].set_title("YZ Slice (Central X)")
    ax_list[2].axis("off")

    # Maximum intensity projections
    ax_list[3].imshow(np.max(volume, axis=0), cmap="gray")
    ax_list[3].set_title("XY Projection (Max)")
    ax_list[3].axis("off")

    ax_list[4].imshow(np.max(volume, axis=1), cmap="gray")
    ax_list[4].set_title("XZ Projection (Max)")
    ax_list[4].axis("off")

    ax_list[5].imshow(np.max(volume, axis=2), cmap="gray")
    ax_list[5].set_title("YZ Projection (Max)")
    ax_list[5].axis("off")

    if title:
        ax_list[0].set_ylabel(title, rotation=0, labelpad=40, ha="right", va="center")


def main():
    """Main function to visualize the dataset."""
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

        # Determine the number of classes
        num_classes = len(samples_by_class)
        logger.info(f"Found {num_classes} classes")

        # Create a figure for visualization
        fig = plt.figure(figsize=(15, 3 * num_classes * args.samples_per_class), constrained_layout=True)
        gs = GridSpec(num_classes * args.samples_per_class, 6, figure=fig)

        # Plot samples for each class
        logger.info(f"Creating visualization with {args.samples_per_class} samples per class...")

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
                # Create axes for this sample
                ax_list = [fig.add_subplot(gs[row, j]) for j in range(6)]

                # Add sample title with index
                title = f"{class_name} ({sample_idx})"

                # Visualize the sample
                visualize_sample(ax_list, sample, title=title)

                row += 1

        # Add overall title
        fig.suptitle(f"Dataset Visualization: {os.path.basename(args.dataset_dir)}", fontsize=16)

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
