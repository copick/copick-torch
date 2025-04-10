"""
Utility functions for parallel processing in copick-torch.
"""

import os
import numpy as np
import torch
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

def extract_and_process_subvolume(args):
    """
    Extract and process a subvolume for parallel processing.
    
    Args:
        args: Tuple of (point, label, tomogram_idx, tomogram_data, boxsize, voxel_spacing)
        
    Returns:
        Tuple of (tensor, label)
    """
    point, label, tomogram_idx, tomogram_data, boxsize, voxel_spacing = args
    
    # Check if tomogram exists
    if tomogram_idx >= len(tomogram_data) or tomogram_data[tomogram_idx] is None:
        raise ValueError(f"No tomogram found at index {tomogram_idx}")
        
    tomogram_zarr = tomogram_data[tomogram_idx]
    
    # Get dimensions of the tomogram
    z_dim, y_dim, x_dim = tomogram_zarr.shape
    
    # Convert coordinates to indices
    x_idx = int(point[0] / voxel_spacing)
    y_idx = int(point[1] / voxel_spacing)
    z_idx = int(point[2] / voxel_spacing)
    
    # Calculate half box sizes
    half_x = boxsize[2] // 2
    half_y = boxsize[1] // 2
    half_z = boxsize[0] // 2
    
    # Calculate bounds with boundary checking
    x_start = max(0, x_idx - half_x)
    x_end = min(x_dim, x_idx + half_x)
    y_start = max(0, y_idx - half_y)
    y_end = min(y_dim, y_idx + half_y)
    z_start = max(0, z_idx - half_z)
    z_end = min(z_dim, z_idx + half_z)
    
    # Extract subvolume
    subvolume = tomogram_zarr[z_start:z_end, y_start:y_end, x_start:x_end]
    
    # Pad if necessary
    if subvolume.shape != boxsize:
        padded = np.zeros(boxsize, dtype=subvolume.dtype)
        
        # Calculate padding dimensions
        pad_z = min(z_end - z_start, boxsize[0])
        pad_y = min(y_end - y_start, boxsize[1])
        pad_x = min(x_end - x_start, boxsize[2])
        
        # Calculate padding offsets (center the subvolume in the padded volume)
        z_offset = (boxsize[0] - pad_z) // 2
        y_offset = (boxsize[1] - pad_y) // 2
        x_offset = (boxsize[2] - pad_x) // 2
        
        # Insert subvolume into padded volume
        padded[
            z_offset:z_offset+pad_z,
            y_offset:y_offset+pad_y,
            x_offset:x_offset+pad_x
        ] = subvolume
        
        subvolume = padded
    
    # Normalize
    if np.std(subvolume) > 0:
        subvolume = (subvolume - np.mean(subvolume)) / np.std(subvolume)
        
    # Add channel dimension and convert to tensor
    subvolume_tensor = torch.as_tensor(subvolume[None, ...], dtype=torch.float32)
    
    return (subvolume_tensor, label)

def batch_normalize(batch, batch_indices):
    """
    Normalize a batch of subvolumes in parallel.
    
    Args:
        batch: List of subvolumes to normalize
        batch_indices: Indices of the subvolumes in the original dataset
        
    Returns:
        List of normalized subvolumes and corresponding indices
    """
    normalized = []
    for i, subvolume in enumerate(batch):
        if np.std(subvolume) > 0:
            subvolume = (subvolume - np.mean(subvolume)) / np.std(subvolume)
        normalized.append((subvolume, batch_indices[i]))
    return normalized

def parallel_preload_data(points, labels, tomogram_indices, tomogram_data, boxsize, voxel_spacing, 
                        num_workers=None, batch_size=32):
    """
    Preload subvolumes in parallel using multiprocessing.
    
    Args:
        points: List of points to extract subvolumes from
        labels: List of labels for each point
        tomogram_indices: List of tomogram indices for each point
        tomogram_data: List of tomogram data
        boxsize: Size of the subvolumes to extract
        voxel_spacing: Voxel spacing
        num_workers: Number of worker processes to use (default: number of CPU cores)
        batch_size: Size of batches to process at once (default: 32)
        
    Returns:
        List of tuples of (tensor, label)
    """
    if num_workers is None:
        # Use number of CPU cores minus 1, with a minimum of 1
        num_workers = max(1, os.cpu_count() - 1)
        
    # Create a partial function with fixed arguments
    extract_func = partial(extract_and_process_subvolume, 
                          tomogram_data=tomogram_data, 
                          boxsize=boxsize, 
                          voxel_spacing=voxel_spacing)
    
    # Prepare arguments for the worker function
    args_list = [(points[i], labels[i], tomogram_indices[i]) for i in range(len(points))]
    
    # Create a process pool
    with mp.Pool(processes=num_workers) as pool:
        # Process data in batches to show progress
        results = []
        for i in tqdm(range(0, len(args_list), batch_size), desc="Processing batches"):
            batch_args = args_list[i:i+batch_size]
            batch_results = pool.map(extract_func, batch_args)
            results.extend(batch_results)
            
    return results
