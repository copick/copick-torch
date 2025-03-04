import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

def extract_subvolume_with_validation(tomogram: np.ndarray, 
                                      center: Tuple[float, float, float], 
                                      size: Tuple[int, int, int],
                                      voxel_size: float) -> Tuple[Optional[np.ndarray], bool, str]:
    """
    Extract a centered subvolume from the tomogram with validation.
    
    Args:
        tomogram: Input tomogram array
        center: Center coordinates (x, y, z) in Angstroms
        size: Size of subvolume to extract (x, y, z) in voxels
        voxel_size: Size of voxel in Angstroms
        
    Returns:
        Tuple of (subvolume or None, is_valid, reason)
    """
    try:
        # Convert center coordinates from Angstroms to voxel coordinates
        center_voxels = [int(round(c / voxel_size)) for c in center]
        
        # Calculate half sizes as integers
        half_size = [s // 2 for s in size]
        
        # Calculate boundaries ensuring integer values
        start = [max(0, c - h) for c, h in zip(center_voxels, half_size)]
        end = [min(s, c + h) for c, h, s in zip(center_voxels, half_size, tomogram.shape)]
        
        # Validate boundaries
        if any(s < 0 for s in start):
            return None, False, f"start coordinates {start} outside bounds"
            
        if any(e >= s for e, s in zip(end, tomogram.shape)):
            return None, False, f"end coordinates {end} outside bounds {tomogram.shape}"
            
        if any(e <= s for s, e in zip(start, end)):
            return None, False, f"invalid slice range: start {start}, end {end}"
        
        # Extract subvolume using explicit integer indices
        subvol = tomogram[
            start[2]:end[2], 
            start[1]:end[1], 
            start[0]:end[0]
        ]
        
        # Verify extracted shape matches requested size
        if subvol.shape != size:
            return None, False, f"extracted shape {subvol.shape} != requested size {size}"
            
        return subvol, True, "valid"
        
    except Exception as e:
        return None, False, f"extraction error: {str(e)}"

def normalize_subvolume(subvolume: np.ndarray) -> np.ndarray:
    """
    Normalize subvolume to zero mean and unit variance.
    
    Args:
        subvolume: Input 3D array
        
    Returns:
        Normalized subvolume
    """
    mean = np.mean(subvolume)
    std = np.std(subvolume)
    return (subvolume - mean) / (std + 1e-6)  # Add epsilon to avoid division by zero

def sample_background_points(tomogram: np.ndarray, 
                          pick_coords: List[List[float]], 
                          box_size: int,
                          min_distance: Optional[int] = None,
                          num_points: int = 50) -> List[np.ndarray]:
    """
    Sample background points that are far from existing picks.
    
    Args:
        tomogram: Input tomogram array
        pick_coords: List of existing pick coordinates in voxels
        box_size: Size of box for volume extraction
        min_distance: Minimum distance from existing picks (defaults to box_size)
        num_points: Number of background points to sample
        
    Returns:
        List of background point coordinates in voxel space
    """
    if min_distance is None:
        min_distance = box_size
        
    pick_coords = np.array(pick_coords) if len(pick_coords) > 0 else np.empty((0, 3))
    
    # Generate random points and filter by distance
    background_points = []
    max_attempts = num_points * 10  # Limit attempts to avoid infinite loop
    attempts = 0
    
    while len(background_points) < num_points and attempts < max_attempts:
        # Generate random point within tomogram bounds
        # Adjust for box size to ensure full boxes can be extracted
        random_point = np.array([
            np.random.randint(box_size//2, tomogram.shape[2] - box_size//2),
            np.random.randint(box_size//2, tomogram.shape[1] - box_size//2),
            np.random.randint(box_size//2, tomogram.shape[0] - box_size//2)
        ])
        
        # Calculate distances to all picks
        if len(pick_coords) > 0:
            distances = np.linalg.norm(pick_coords - random_point, axis=1)
            if np.min(distances) >= min_distance:
                background_points.append(random_point)
        else:
            # If no picks exist, accept all points
            background_points.append(random_point)
        
        attempts += 1
        
    return background_points