#!/usr/bin/env python3
"""
Script to create, save, and load a particle dataset from a CZ cryoET Dataset.
This uses the MinimalCopickDataset class with the enhanced save/load functionality.
"""

import os
import argparse
import logging
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from copick_torch import MinimalCopickDataset
from copick_torch.logging import setup_logging

def parse_arguments():
    parser = argparse.ArgumentParser(description='Create a particle dataset from a coPICK dataset')
    parser.add_argument('--dataset_id', type=int, required=True, help='Dataset ID from coPICK')
    parser.add_argument('--overlay_root', type=str, default='/tmp/copick', help='Overlay root directory')
    parser.add_argument('--voxel_spacing', type=float, default=10.012, help='Voxel spacing for coordinate conversion')
    parser.add_argument('--cube_size', type=int, default=48, help='Size of the cubic subvolume in voxels')
    parser.add_argument('--preload', action='store_true', help='Whether to preload all subvolumes into memory')
    parser.add_argument('--output_dir', type=str, help='Directory to save the dataset (default: saved_particle_dataset_<dataset_id>)')
    parser.add_argument('--include_background', action='store_true', help='Whether to include background samples')
    parser.add_argument('--background_ratio', type=float, default=0.2, help='Ratio of background to particle samples')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    
    args = parser.parse_args()
    
    # Set default output directory if not provided
    if args.output_dir is None:
        args.output_dir = f"saved_particle_dataset_{args.dataset_id}"
    
    return args

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    # Dataset loading parameters
    dataset_id = args.dataset_id
    overlay_root = args.overlay_root
    voxel_spacing = args.voxel_spacing
    cube_size = args.cube_size
    
    # Create dataset
    logger.info(f"Creating dataset from dataset ID {dataset_id} with voxel spacing {voxel_spacing}...")
    dataset = MinimalCopickDataset(
        dataset_id=dataset_id,
        overlay_root=overlay_root,
        boxsize=(cube_size, cube_size, cube_size),
        voxel_spacing=voxel_spacing,
        include_background=args.include_background,
        background_ratio=args.background_ratio,
        max_samples=args.max_samples,
        preload=args.preload
    )
    
    logger.info(f"Dataset created with {len(dataset)} samples")
    
    # Print class distribution
    class_distribution = dataset.get_class_distribution()
    logger.info("Class distribution:")
    for class_name, count in class_distribution.items():
        logger.info(f"  {class_name}: {count} samples")
    
    # Test loading a few samples
    logger.info("Testing data loading...")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))
    volumes, labels = batch
    
    logger.info(f"Loaded batch with shape: {volumes.shape}, labels: {labels}")
    
    # Save the dataset to disk
    save_dir = args.output_dir
    logger.info(f"Saving dataset to {save_dir}...")
    dataset.save(save_dir)
    
    # Test loading the saved dataset
    logger.info(f"Testing dataset loading from {save_dir}...")
    
    if args.preload:
        # For preloaded datasets, we don't need to provide tomogram_data
        loaded_dataset = MinimalCopickDataset.load(save_dir)
    else:
        logger.info("This dataset was not preloaded, so we can't fully load it without the original tomogram data")
        logger.info("In a real application, you would need to provide the tomogram_data parameter")
        logger.info("For demonstration, we'll try loading the metadata...")
        
        try:
            # Just check if the metadata loads correctly
            import json
            with open(os.path.join(save_dir, 'metadata.json'), 'r') as f:
                metadata = json.load(f)
            logger.info(f"Successfully loaded metadata: {metadata}")
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
    
    # Print confirmation
    logger.info("Dataset creation and saving completed successfully!")

if __name__ == "__main__":
    main()
