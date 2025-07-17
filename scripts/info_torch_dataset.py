#!/usr/bin/env python
"""
Utility script to load a saved MinimalCopickDataset and display information about it.

Usage:
    python info_torch_dataset.py --input_dir /path/to/saved/dataset
"""

import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

from copick_torch.minimal_dataset import MinimalCopickDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Load and display information about a saved MinimalCopickDataset")

    # Required arguments
    parser.add_argument("--input_dir", type=str, required=True, help="Directory where the dataset is saved")

    # Optional arguments
    parser.add_argument(
        "--output_pdf",
        type=str,
        default=None,
        help="Path to save visualization PDF (default: input_dir/dataset_overview.pdf)",
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=3,
        help="Number of sample visualizations per class (default: 3)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def visualize_volume(volume, class_name, sample_idx, subplot, cmap="gray"):
    """Visualize a 3D volume as a montage of slices."""
    # Get dimensions
    if isinstance(volume, torch.Tensor):
        volume = volume.squeeze().numpy()

    # Create a montage of slices
    depth = volume.shape[0]
    n_slices = min(5, depth)  # Show up to 5 slices
    slice_indices = [int(i * depth / n_slices) for i in range(n_slices)]

    for i, z in enumerate(slice_indices):
        ax = subplot[i]
        ax.imshow(volume[z], cmap=cmap)
        ax.set_title(f"Slice {z}")
        ax.axis("off")

    # Set common title
    subplot[0].figure.suptitle(f"{class_name} - Sample {sample_idx}")


def main():
    """Main function to load and display dataset information."""
    # Parse command line arguments
    args = parse_args()

    # Set log level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check if input directory exists
    if not os.path.isdir(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return 1

    # Check if metadata file exists
    metadata_path = os.path.join(args.input_dir, "metadata.json")
    if not os.path.isfile(metadata_path):
        logger.error(f"Metadata file not found: {metadata_path}")
        return 1

    # Check if we have preloaded tensors
    subvolumes_path = os.path.join(args.input_dir, "subvolumes.pt")
    labels_path = os.path.join(args.input_dir, "labels.pt")

    has_preloaded_tensors = os.path.exists(subvolumes_path) and os.path.exists(labels_path)
    if has_preloaded_tensors:
        logger.info("Detected preloaded tensor data.")
    else:
        logger.warning("No preloaded tensor data found. Dataset was saved without preloading.")

    try:
        # Load metadata file to display basic information
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        logger.info("Dataset Metadata:")
        logger.info(f"  Dataset ID: {metadata.get('dataset_id')}")
        logger.info(f"  Box size: {metadata.get('boxsize')}")
        logger.info(f"  Voxel spacing: {metadata.get('voxel_spacing')}")
        logger.info(f"  Include background: {metadata.get('include_background')}")
        logger.info(f"  Background ratio: {metadata.get('background_ratio')}")
        logger.info(f"  Min background distance: {metadata.get('min_background_distance')}")
        logger.info(f"  Preload: {metadata.get('preload', False)}")

        # Display class mapping
        name_to_label = metadata.get("name_to_label", {})
        logger.info("Class mapping:")
        for name, label in name_to_label.items():
            logger.info(f"  {name}: {label}")

        # Calculate class distribution
        class_counts = {}

        # If we have preloaded tensors, analyze them
        if has_preloaded_tensors:
            logger.info("Analyzing preloaded tensors...")

            # Load the labels tensor to get class distribution
            labels = torch.load(labels_path)

            for label in labels:
                label_val = label.item()

                # Handle background label
                if label_val == -1:
                    class_name = "background"
                else:
                    # Find class name for this label
                    class_name = None
                    for name, idx in name_to_label.items():
                        if idx == label_val:
                            class_name = name
                            break

                    if class_name is None:
                        class_name = f"Unknown_Label_{label_val}"

                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1

            logger.info(f"Tensor data shape: {torch.load(subvolumes_path).shape}")
        else:
            # Otherwise, load sample information
            samples_path = os.path.join(args.input_dir, "samples.json")
            if os.path.isfile(samples_path):
                with open(samples_path, "r") as f:
                    samples = json.load(f)

                logger.info("Analyzing sample information...")

                # Calculate class distribution from sample data
                for sample in samples:
                    label = sample["label"]

                    # Handle background label
                    if label == -1:
                        class_name = "background"
                    else:
                        # Find class name for this label
                        class_name = None
                        for name, idx in name_to_label.items():
                            if idx == label:
                                class_name = name
                                break

                        if class_name is None:
                            class_name = f"Unknown_Label_{label}"

                    if class_name not in class_counts:
                        class_counts[class_name] = 0
                    class_counts[class_name] += 1

        # Display class distribution
        total_samples = sum(class_counts.values())
        logger.info(f"Total samples: {total_samples}")
        logger.info("Class distribution:")
        for class_name, count in class_counts.items():
            percentage = 100 * count / total_samples
            logger.info(f"  {class_name}: {count} samples ({percentage:.2f}%)")

        # Load tomogram information
        tomogram_path = os.path.join(args.input_dir, "tomogram_info.json")
        if os.path.isfile(tomogram_path):
            with open(tomogram_path, "r") as f:
                tomogram_info = json.load(f)
            logger.info(f"Number of tomograms: {len(tomogram_info)}")
            for idx, tomogram in enumerate(tomogram_info):
                logger.info(f"  Tomogram {idx}: shape {tomogram.get('shape')}")

        # Load the dataset
        logger.info(f"Loading dataset from {args.input_dir}...")
        dataset = MinimalCopickDataset.load(args.input_dir)
        logger.info(f"Dataset loaded with {len(dataset)} samples")

        # Get a sample to check dimensions
        if len(dataset) > 0:
            sample_volume, sample_label = dataset[0]
            logger.info(f"Sample volume shape: {sample_volume.shape}")
            logger.info(f"Sample label: {sample_label}")

        # Check if we should generate visualizations
        if args.output_pdf is None:
            output_pdf = os.path.join(args.input_dir, "dataset_overview.pdf")
        else:
            output_pdf = args.output_pdf

        # Generate class distribution and sample visualizations
        with PdfPages(output_pdf) as pdf:
            # Class distribution visualization
            plt.figure(figsize=(10, 6))
            classes = list(class_counts.keys())
            counts = [class_counts[c] for c in classes]

            plt.bar(classes, counts)
            plt.xlabel("Class")
            plt.ylabel("Number of Samples")
            plt.title("Class Distribution")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # Sample visualizations
            class_indices = {}  # Dictionary to track indices for each class

            # Initialize class indices
            for i, (vol, label) in enumerate(dataset):
                class_name = (
                    "background"
                    if label == -1
                    else next((name for name, idx in name_to_label.items() if idx == label), f"Unknown_{label}")
                )
                if class_name not in class_indices:
                    class_indices[class_name] = []

                # Store a limited number of indices per class
                if len(class_indices[class_name]) < args.samples_per_class:
                    class_indices[class_name].append(i)

            # Generate visualizations for each class
            for class_name, indices in class_indices.items():
                for i, idx in enumerate(indices):
                    vol, _ = dataset[idx]

                    # Create a figure with subplots for slices
                    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
                    visualize_volume(vol, class_name, i + 1, axes)
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

            logger.info(f"Visualizations saved to {output_pdf}")

    except Exception as e:
        logger.exception(f"Error loading or analyzing dataset: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
