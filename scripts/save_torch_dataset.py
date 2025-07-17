#!/usr/bin/env python
"""
Utility script to create and save a MinimalCopickDataset to disk.

Usage:
    python save_torch_dataset.py --dataset_id 10440 --output_dir /path/to/output
"""

import argparse
import logging
import os

import copick
from tqdm import tqdm

from copick_torch.minimal_dataset import MinimalCopickDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create and save a MinimalCopickDataset to disk")

    # Required arguments
    parser.add_argument("--dataset_id", type=int, required=True, help="Dataset ID from the CZ cryoET Data Portal")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the dataset")

    # Optional arguments
    parser.add_argument(
        "--overlay_root",
        type=str,
        default="/tmp/copick_overlay",
        help="Root directory for the overlay storage (default: /tmp/copick_overlay)",
    )
    parser.add_argument(
        "--boxsize",
        type=int,
        nargs=3,
        default=[48, 48, 48],
        help="Size of subvolumes to extract as z y x (default: 48 48 48)",
    )
    parser.add_argument("--voxel_spacing", type=float, default=10.012, help="Voxel spacing to use (default: 10.012)")
    parser.add_argument("--include_background", action="store_true", help="Include background samples in the dataset")
    parser.add_argument(
        "--background_ratio",
        type=float,
        default=0.2,
        help="Ratio of background to particle samples (default: 0.2)",
    )
    parser.add_argument(
        "--min_background_distance",
        type=float,
        default=None,
        help="Minimum distance from particles for background (default: max boxsize)",
    )
    parser.add_argument(
        "--no-preload",
        dest="preload",
        action="store_false",
        help="Disable preloading tensors (not recommended)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # Set defaults
    parser.set_defaults(preload=True)

    return parser.parse_args()


def main():
    """Main function to create and save the dataset."""
    # Parse command line arguments
    args = parse_args()

    # Set log level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Log the parameters
    logger.info("Creating dataset with the following parameters:")
    logger.info(f"  Dataset ID: {args.dataset_id}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Overlay root: {args.overlay_root}")
    logger.info(f"  Box size: {args.boxsize}")
    logger.info(f"  Voxel spacing: {args.voxel_spacing}")
    logger.info(f"  Include background: {args.include_background}")
    logger.info(f"  Background ratio: {args.background_ratio}")
    logger.info(f"  Min background distance: {args.min_background_distance}")
    logger.info(f"  Preload: {args.preload}")

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Load the dataset from CoPICK
        logger.info(f"Loading dataset {args.dataset_id} from CoPICK...")
        proj = copick.from_czcdp_datasets([args.dataset_id], overlay_root=args.overlay_root)

        # Create the dataset
        logger.info("Creating MinimalCopickDataset...")
        if args.preload:
            logger.info("With preloading (this saves all subvolumes as tensor data)")
        else:
            logger.info("Without preloading (this saves only metadata and coordinates)")

        dataset = MinimalCopickDataset(
            proj=proj,
            boxsize=tuple(args.boxsize),
            voxel_spacing=args.voxel_spacing,
            include_background=args.include_background,
            background_ratio=args.background_ratio,
            min_background_distance=args.min_background_distance,
            preload=args.preload,
        )

        # Save the dataset
        logger.info(f"Saving dataset to {args.output_dir}...")
        dataset.save(args.output_dir)

        # Log completion
        logger.info(f"Dataset saved successfully with {len(dataset)} samples.")

        # Print class distribution
        distribution = dataset.get_class_distribution()
        logger.info("Class distribution:")
        for class_name, count in distribution.items():
            logger.info(f"  {class_name}: {count} samples")

    except Exception as e:
        logger.exception(f"Error creating or saving dataset: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
