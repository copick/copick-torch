"""
Extension to MinimalCopickDataset that adds preloading capability.
"""

import numpy as np
import torch
import os
import logging
from tqdm import tqdm
from copick_torch.minimal_dataset import MinimalCopickDataset

logger = logging.getLogger(__name__)

class PreloadedCopickDataset(MinimalCopickDataset):
    """
    Extension of MinimalCopickDataset that preloads all data into memory.
    
    Unlike the base MinimalCopickDataset, this implementation:
    1. Preloads all subvolumes into memory during initialization
    2. Stores the actual tensors when saving to disk
    3. Can directly load the tensors from disk without needing tomogram data
    """
    
    def __init__(
        self,
        proj=None,
        dataset_id=None,
        overlay_root=None,
        boxsize=(48, 48, 48),
        voxel_spacing=10.012,
        include_background=False,
        background_ratio=0.2,
        min_background_distance=None
    ):
        """
        Initialize a PreloadedCopickDataset.
        
        Args:
            proj: A copick project object. If provided, dataset_id and overlay_root are ignored.
            dataset_id: Dataset ID from the CZ cryoET Data Portal. Only used if proj is None.
            overlay_root: Root directory for the overlay storage. Only used if proj is None.
            boxsize: Size of the subvolumes to extract (z, y, x)
            voxel_spacing: Voxel spacing to use for extraction
            include_background: Whether to include background samples
            background_ratio: Ratio of background to particle samples
            min_background_distance: Minimum distance from particles for background samples
        """
        # Initialize the base class
        super().__init__(
            proj=proj,
            dataset_id=dataset_id,
            overlay_root=overlay_root,
            boxsize=boxsize,
            voxel_spacing=voxel_spacing,
            include_background=include_background,
            background_ratio=background_ratio,
            min_background_distance=min_background_distance
        )
        
        # Preload the data
        if hasattr(self, '_points') and len(self._points) > 0:
            self._preload_data()
    
    def _preload_data(self):
        """Preload all subvolumes into memory."""
        logger.info(f"Preloading {len(self._points)} subvolumes into memory...")
        
        # Initialize storage for preloaded data
        self._subvolumes = []
        
        # Extract and store all subvolumes
        for idx in tqdm(range(len(self._points))):
            point = self._points[idx]
            label = self._labels[idx]
            tomogram_idx = self._tomogram_indices[idx] if hasattr(self, '_tomogram_indices') else 0
            
            # Extract the subvolume
            subvolume = self.extract_subvolume(point, tomogram_idx)
            
            # Normalize
            if np.std(subvolume) > 0:
                subvolume = (subvolume - np.mean(subvolume)) / np.std(subvolume)
                
            # Add channel dimension and convert to tensor
            subvolume_tensor = torch.as_tensor(subvolume[None, ...], dtype=torch.float32)
            
            # Store the tensor with its label
            self._subvolumes.append((subvolume_tensor, label))
        
        logger.info(f"Preloaded {len(self._subvolumes)} subvolumes")
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (subvolume, label)
        """
        # If data is preloaded, return from preloaded data
        if hasattr(self, '_subvolumes') and self._subvolumes:
            return self._subvolumes[idx]
        
        # Otherwise, fall back to parent class implementation
        return super().__getitem__(idx)
    
    def save(self, save_dir):
        """
        Save the dataset to disk for later reloading.
        
        This implementation saves the actual preloaded tensors.
        
        Args:
            save_dir: Directory to save the dataset
        """
        # Ensure we have preloaded data
        if not hasattr(self, '_subvolumes') or not self._subvolumes:
            logger.info("Data not preloaded yet, preloading now...")
            self._preload_data()
        
        # Create the output directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metadata (same as parent class)
        super().save(save_dir)
        
        # Extract tensors and labels
        subvolumes = []
        labels = []
        
        for volume, label in self._subvolumes:
            subvolumes.append(volume)
            labels.append(label)
        
        # Stack tensors into a single tensor
        subvolumes_tensor = torch.stack(subvolumes)
        labels_tensor = torch.tensor(labels)
        
        # Save tensors to disk
        torch.save(subvolumes_tensor, os.path.join(save_dir, 'subvolumes.pt'))
        torch.save(labels_tensor, os.path.join(save_dir, 'labels.pt'))
        
        logger.info(f"Saved preloaded tensors to {save_dir}")
    
    @classmethod
    def load(cls, save_dir, proj=None):
        """
        Load a previously saved dataset.
        
        This implementation can load the preloaded tensors directly.
        
        Args:
            save_dir: Directory where the dataset was saved
            proj: Optional copick project object (not required for preloaded data)
            
        Returns:
            Loaded PreloadedCopickDataset instance
        """
        # Check if we have preloaded tensors
        subvolumes_path = os.path.join(save_dir, 'subvolumes.pt')
        labels_path = os.path.join(save_dir, 'labels.pt')
        
        if os.path.exists(subvolumes_path) and os.path.exists(labels_path):
            # Create a minimal dataset instance
            dataset = super().load(save_dir, proj)
            
            # Load preloaded tensors
            logger.info(f"Loading preloaded tensors from {save_dir}...")
            subvolumes = torch.load(subvolumes_path)
            labels = torch.load(labels_path)
            
            # Convert to the format used by the class
            dataset._subvolumes = [(subvolumes[i], labels[i].item()) for i in range(len(labels))]
            
            logger.info(f"Loaded {len(dataset._subvolumes)} preloaded subvolumes")
            return dataset
        else:
            # If we don't have preloaded tensors, use parent class loading
            # but then preload the data
            dataset = super().load(save_dir, proj)
            
            # Preload the data
            if hasattr(dataset, '_points') and len(dataset._points) > 0:
                dataset._preload_data()
                
            return dataset
