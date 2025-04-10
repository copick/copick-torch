#!/usr/bin/env python3
"""
Script to load a previously saved particle dataset and use it for training or inference.
"""

import os
import argparse
import logging
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from copick_torch import MinimalCopickDataset
from copick_torch.logging import setup_logging

def parse_arguments():
    parser = argparse.ArgumentParser(description='Load a previously saved particle dataset')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Directory where the dataset was saved')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for data loading')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for data loading')
    parser.add_argument('--visualize', action='store_true', help='Whether to visualize sample subvolumes')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    
    return parser.parse_args()

def visualize_subvolumes(volumes, labels, class_names, num_samples=4):
    """Visualize a few sample subvolumes from the dataset"""
    num_samples = min(num_samples, volumes.shape[0])
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i in range(num_samples):
        # Get the volume and label
        volume = volumes[i, 0].numpy()  # Remove channel dimension
        label = labels[i].item()
        
        # Get class name
        if label == -1:
            class_name = "background"
        else:
            class_names_list = list(class_names)
            class_name = class_names_list[label] if label < len(class_names_list) else f"Unknown ({label})"
        
        # Get middle slices
        z_mid = volume.shape[0] // 2
        y_mid = volume.shape[1] // 2
        x_mid = volume.shape[2] // 2
        
        # Plot XY slice
        axes[i, 0].imshow(volume[z_mid, :, :], cmap='gray')
        axes[i, 0].set_title(f"XY Slice (class: {class_name})")
        
        # Plot YZ slice
        axes[i, 1].imshow(volume[:, :, x_mid], cmap='gray')
        axes[i, 1].set_title(f"YZ Slice")
        
        # Plot XZ slice
        axes[i, 2].imshow(volume[:, y_mid, :], cmap='gray')
        axes[i, 2].set_title(f"XZ Slice")
    
    plt.tight_layout()
    plt.savefig("sample_subvolumes.png")
    print(f"Visualization saved to sample_subvolumes.png")

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load the dataset
    logger.info(f"Loading dataset from {args.dataset_dir}...")
    
    try:
        dataset = MinimalCopickDataset.load(args.dataset_dir)
        logger.info(f"Successfully loaded dataset with {len(dataset)} samples")
        
        # Print class distribution
        class_distribution = dataset.get_class_distribution()
        logger.info("Class distribution:")
        for class_name, count in class_distribution.items():
            logger.info(f"  {class_name}: {count} samples")
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers
        )
        
        # Iterate through a few batches to demonstrate usage
        logger.info(f"Iterating through {min(5, len(dataloader))} batches...")
        
        for i, (volumes, labels) in enumerate(tqdm(dataloader)):
            # Simulate some processing
            logger.debug(f"Batch {i+1}: volumes shape = {volumes.shape}, labels shape = {labels.shape}")
            
            # Visualize the first batch if requested
            if i == 0 and args.visualize:
                logger.info("Visualizing sample subvolumes...")
                visualize_subvolumes(volumes, labels, dataset.keys())
            
            # Stop after a few batches for demonstration
            if i >= 4:
                break
                
        logger.info("Dataset loading and iteration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        logger.error("If the dataset was not preloaded, you need to provide the original tomogram data")
        logger.error("This demonstration script only works with preloaded datasets.")

if __name__ == "__main__":
    main()
