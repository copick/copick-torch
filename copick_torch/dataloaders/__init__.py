"""
CryoET Data Portal dataloader for copick.

This module provides dataloaders specifically designed for loading data from the CryoET Data Portal,
with automatic rescaling to a target resolution.
"""

import os
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any, Sequence

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom

import copick
from monai.transforms import Compose, ScaleIntensity, EnsureChannelFirst, ToTensor


class CryoETDataPortalDataset(Dataset):
    """
    A PyTorch dataset for working with CryoET data from the CZ CryoET Data Portal.
    
    This dataset automatically rescales tomograms to the target resolution using scipy.ndimage.zoom.
    
    Args:
        dataset_ids: list of dataset IDs from the CryoET Data Portal 
        overlay_root: root URL for the overlay storage
        boxsize: size of extracted boxes (z, y, x)
        voxel_spacing: target voxel spacing in Ångstroms
        transform: transforms to apply to the tomogram
        cache_dir: directory to cache rescaled tomograms
        use_cache: whether to use cached tomograms if available
        batch_size: batch size for returning data
        shuffle: whether to shuffle the order of tomograms
        num_workers: number of worker processes for data loading
    """
    
    def __init__(
        self,
        dataset_ids: List[int],
        overlay_root: str,
        boxsize: Tuple[int, int, int] = (32, 32, 32),
        voxel_spacing: float = 10.0,
        transform: Optional[Any] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0
    ):
        self.dataset_ids = dataset_ids
        self.overlay_root = overlay_root
        self.boxsize = boxsize
        self.voxel_spacing = voxel_spacing
        self.transform = transform if transform is not None else Compose([
            EnsureChannelFirst(),
            ScaleIntensity(),
            ToTensor()
        ])
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        self.logger = logging.getLogger(__name__)
        
        # Create cache directory if necessary
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize copick project
        self.logger.info(f"Initializing copick project for dataset ids: {dataset_ids}")
        self.root = copick.from_czcdp_datasets(
            dataset_ids=dataset_ids,
            overlay_root=overlay_root
        )
        
        # Collect all runs and tomograms
        self.runs = self.root.runs
        self.tomograms = []
        self.tomogram_metadata = []
        
        for run in self.runs:
            # Get the closest voxel spacing available
            available_spacings = [vs.voxel_size for vs in run.voxel_spacings]
            closest_spacing = min(available_spacings, key=lambda x: abs(x - voxel_spacing))
            
            vs = run.get_voxel_spacing(closest_spacing)
            if vs is None:
                continue
                
            # Get all tomograms for this voxel spacing
            for tomo in vs.tomograms:
                self.tomograms.append(tomo)
                self.tomogram_metadata.append({
                    'run': run,
                    'voxel_spacing': vs,
                    'original_spacing': closest_spacing,
                    'target_spacing': voxel_spacing
                })
        
        self.logger.info(f"Found {len(self.tomograms)} tomograms in {len(self.runs)} runs")
    
    def __len__(self):
        return len(self.tomograms)
    
    def __getitem__(self, idx):
        tomogram = self.tomograms[idx]
        metadata = self.tomogram_metadata[idx]
        
        # Check if rescaled tomogram is cached
        cache_path = None
        if self.cache_dir is not None:
            run_name = metadata['run'].name
            vs_value = metadata['original_spacing']
            target_vs = metadata['target_spacing']
            tomo_type = tomogram.tomo_type
            cache_filename = f"{run_name}_{vs_value:.2f}_to_{target_vs:.2f}_{tomo_type}.npy"
            cache_path = Path(self.cache_dir) / cache_filename
        
        # Load from cache if available and requested
        if cache_path is not None and cache_path.exists() and self.use_cache:
            self.logger.info(f"Loading rescaled tomogram from cache: {cache_path}")
            try:
                rescaled_tomo = np.load(cache_path)
            except Exception as e:
                self.logger.error(f"Failed to load from cache: {e}")
                rescaled_tomo = self._load_and_rescale_tomogram(tomogram, metadata)
        else:
            # Load and rescale tomogram
            rescaled_tomo = self._load_and_rescale_tomogram(tomogram, metadata)
            
            # Cache the rescaled tomogram if requested
            if cache_path is not None and self.use_cache:
                self.logger.info(f"Caching rescaled tomogram: {cache_path}")
                try:
                    np.save(cache_path, rescaled_tomo)
                except Exception as e:
                    self.logger.error(f"Failed to save to cache: {e}")
        
        # Apply transforms
        if self.transform:
            rescaled_tomo = self.transform(rescaled_tomo)
        
        return rescaled_tomo, {'idx': idx, 'metadata': metadata}
    
    def _load_and_rescale_tomogram(self, tomogram, metadata):
        """
        Load and rescale a tomogram to the target voxel spacing.
        
        Args:
            tomogram: tomogram object from copick
            metadata: metadata dictionary
            
        Returns:
            rescaled_tomo: numpy array containing the rescaled tomogram
        """
        self.logger.info(f"Loading tomogram for run {metadata['run'].name}")
        
        # Load the tomogram
        tomo_array = tomogram.numpy()
        
        # Calculate zoom factors for rescaling
        original_spacing = metadata['original_spacing']
        target_spacing = metadata['target_spacing']
        zoom_factors = [original_spacing / target_spacing] * 3
        
        # Skip rescaling if spacing is already very close
        if abs(original_spacing - target_spacing) < 0.01:
            self.logger.info(f"Skipping rescaling as spacing is already close: {original_spacing:.2f}")
            return tomo_array
        
        # Rescale the tomogram
        self.logger.info(f"Rescaling tomogram from {original_spacing:.2f}Å to {target_spacing:.2f}Å")
        self.logger.info(f"Tomogram shape before rescaling: {tomo_array.shape}")
        
        # Use scipy.ndimage.zoom for rescaling
        try:
            rescaled_tomo = zoom(tomo_array, zoom_factors, order=1, mode='constant')
            self.logger.info(f"Tomogram shape after rescaling: {rescaled_tomo.shape}")
            return rescaled_tomo
        except Exception as e:
            self.logger.error(f"Failed to rescale tomogram: {e}")
            # Return original if rescaling fails
            return tomo_array
    
    def get_dataloader(self):
        """
        Get a DataLoader for this dataset.
        
        Returns:
            DataLoader: PyTorch DataLoader
        """
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )
    
    def get_picks_for_tomogram(self, idx):
        """
        Get particle picks for a specific tomogram.
        
        Args:
            idx: index of the tomogram
            
        Returns:
            list of particle coordinates (rescaled to target resolution)
        """
        if idx < 0 or idx >= len(self.tomograms):
            raise IndexError(f"Index {idx} out of range for tomogram list of length {len(self.tomograms)}")
        
        metadata = self.tomogram_metadata[idx]
        run = metadata['run']
        
        # Get all picks for this run
        picks = run.get_picks()
        
        # Initialize list to hold all coordinates
        all_coords = []
        
        # Process each pick set
        for pick in picks:
            try:
                # Convert picks to numpy coordinates
                points, _ = pick.numpy()
                
                # Rescale coordinates from original to target resolution
                original_spacing = metadata['original_spacing']
                target_spacing = metadata['target_spacing']
                scale_factor = original_spacing / target_spacing
                
                # Scale coordinates
                points = points * scale_factor
                
                # Add to list
                all_coords.append(points)
            except Exception as e:
                self.logger.error(f"Error processing picks: {e}")
        
        # Combine all coordinates
        if all_coords:
            return np.vstack(all_coords)
        else:
            return np.zeros((0, 3))
    
    def get_all_picks(self):
        """
        Get all particle picks for all tomograms.
        
        Returns:
            dict: mapping from tomogram idx to particle coordinates
        """
        all_picks = {}
        for idx in range(len(self.tomograms)):
            all_picks[idx] = self.get_picks_for_tomogram(idx)
        return all_picks


class CryoETParticleDataset(Dataset):
    """
    A PyTorch dataset for particle picking in CryoET data from the CZ CryoET Data Portal.
    
    This dataset extracts subvolumes around particle coordinates for training detection models.
    
    Args:
        dataset_ids: list of dataset IDs from the CryoET Data Portal 
        overlay_root: root URL for the overlay storage
        boxsize: size of extracted boxes (z, y, x)
        voxel_spacing: target voxel spacing in Ångstroms
        include_background: whether to include background (non-particle) samples
        background_ratio: ratio of background samples to particle samples
        min_background_distance: minimum distance from particles for background samples
        transform: transforms to apply to the extracted subvolumes
        cache_dir: directory to cache rescaled tomograms
        use_cache: whether to use cached tomograms if available
    """
    
    def __init__(
        self,
        dataset_ids: List[int],
        overlay_root: str,
        boxsize: Tuple[int, int, int] = (32, 32, 32),
        voxel_spacing: float = 10.0,
        include_background: bool = True,
        background_ratio: float = 0.5,
        min_background_distance: float = 20.0,
        transform: Optional[Any] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True
    ):
        self.dataset_ids = dataset_ids
        self.overlay_root = overlay_root
        self.boxsize = boxsize
        self.voxel_spacing = voxel_spacing
        self.include_background = include_background
        self.background_ratio = background_ratio
        self.min_background_distance = min_background_distance
        self.transform = transform if transform is not None else Compose([
            EnsureChannelFirst(),
            ScaleIntensity(),
            ToTensor()
        ])
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        
        self.logger = logging.getLogger(__name__)
        
        # Create the base dataset to load and rescale tomograms
        self.base_dataset = CryoETDataPortalDataset(
            dataset_ids=dataset_ids,
            overlay_root=overlay_root,
            boxsize=boxsize,
            voxel_spacing=voxel_spacing,
            transform=None,  # We'll apply transforms later
            cache_dir=cache_dir,
            use_cache=use_cache
        )
        
        # Load all tomograms and picks
        self.tomograms = []
        self.particle_coords = []
        self.background_coords = []
        
        # Flag to track if dataset is fully initialized
        self._initialized = False
        
    def initialize(self):
        """
        Fully initialize the dataset by loading all tomograms and extracting particles.
        This is separated from __init__ to allow lazy loading.
        """
        if self._initialized:
            return
        
        # Load all tomograms
        for idx in range(len(self.base_dataset)):
            # Load tomogram
            tomo, metadata = self.base_dataset[idx]
            self.tomograms.append(tomo)
            
            # Get particle picks for this tomogram
            particle_coords = self.base_dataset.get_picks_for_tomogram(idx)
            self.particle_coords.append(particle_coords)
            
            # Generate background samples
            if self.include_background:
                bg_coords = self._generate_background_coords(tomo, particle_coords)
                self.background_coords.append(bg_coords)
            else:
                self.background_coords.append(np.zeros((0, 3)))
        
        # Set up indices for accessing particles and background
        self.particle_indices = []
        self.background_indices = []
        
        for tomo_idx, (particles, backgrounds) in enumerate(zip(self.particle_coords, self.background_coords)):
            for particle_idx in range(len(particles)):
                self.particle_indices.append((tomo_idx, particle_idx, True))  # True = particle
            
            for bg_idx in range(len(backgrounds)):
                self.background_indices.append((tomo_idx, bg_idx, False))  # False = background
        
        self._initialized = True
        
        self.logger.info(f"Dataset initialized with {len(self.particle_indices)} particles and {len(self.background_indices)} background samples")
        
    def _generate_background_coords(self, tomogram, particle_coords):
        """
        Generate random background coordinates away from particles.
        
        Args:
            tomogram: tomogram as numpy array
            particle_coords: array of particle coordinates
            
        Returns:
            background_coords: array of background coordinates
        """
        if len(particle_coords) == 0:
            return np.zeros((0, 3))
        
        # Calculate number of background samples to generate
        num_particles = len(particle_coords)
        num_backgrounds = int(num_particles * self.background_ratio)
        
        # Generate random coordinates
        background_coords = []
        max_attempts = num_backgrounds * 10  # Limit attempts to avoid infinite loop
        
        # Get tomogram shape
        z_max, y_max, x_max = tomogram.shape
        half_box = np.array(self.boxsize) // 2
        
        attempts = 0
        while len(background_coords) < num_backgrounds and attempts < max_attempts:
            # Generate random coordinates within valid range
            z = np.random.randint(half_box[0], z_max - half_box[0])
            y = np.random.randint(half_box[1], y_max - half_box[1])
            x = np.random.randint(half_box[2], x_max - half_box[2])
            
            coord = np.array([z, y, x])
            
            # Check distance to all particles
            if len(particle_coords) > 0:
                distances = np.sqrt(np.sum((particle_coords - coord)**2, axis=1))
                min_distance = np.min(distances)
                
                # Only accept if far enough from all particles
                if min_distance >= self.min_background_distance:
                    background_coords.append(coord)
            else:
                # If no particles, just accept the coordinate
                background_coords.append(coord)
            
            attempts += 1
        
        return np.array(background_coords)
    
    def __len__(self):
        # Make sure dataset is initialized
        if not self._initialized:
            self.initialize()
        
        return len(self.particle_indices) + len(self.background_indices)
    
    def __getitem__(self, idx):
        # Make sure dataset is initialized
        if not self._initialized:
            self.initialize()
        
        # Determine if this is a particle or background sample
        if idx < len(self.particle_indices):
            tomo_idx, coord_idx, is_particle = self.particle_indices[idx]
            coords = self.particle_coords[tomo_idx][coord_idx]
        else:
            # Adjust index for background samples
            bg_idx = idx - len(self.particle_indices)
            tomo_idx, coord_idx, is_particle = self.background_indices[bg_idx]
            coords = self.background_coords[tomo_idx][coord_idx]
        
        # Get tomogram
        tomogram = self.tomograms[tomo_idx]
        
        # Extract subvolume centered at coordinates
        subvolume = self._extract_subvolume(tomogram, coords)
        
        # Create target label (1 for particle, 0 for background)
        label = 1 if is_particle else 0
        
        # Apply transforms
        if self.transform:
            subvolume = self.transform(subvolume)
        
        return subvolume, label
    
    def _extract_subvolume(self, tomogram, coords):
        """
        Extract a subvolume from the tomogram centered at the given coordinates.
        
        Args:
            tomogram: tomogram as numpy array
            coords: coordinates (z, y, x) of the center point
            
        Returns:
            subvolume: extracted subvolume
        """
        # Get tomogram shape
        z_max, y_max, x_max = tomogram.shape
        
        # Calculate half box size
        half_box = np.array(self.boxsize) // 2
        
        # Calculate extraction ranges
        z_start = max(0, int(coords[0] - half_box[0]))
        z_end = min(z_max, int(coords[0] + half_box[0]))
        y_start = max(0, int(coords[1] - half_box[1]))
        y_end = min(y_max, int(coords[1] + half_box[1]))
        x_start = max(0, int(coords[2] - half_box[2]))
        x_end = min(x_max, int(coords[2] + half_box[2]))
        
        # Extract subvolume
        subvolume = tomogram[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Handle case where extracted subvolume is smaller than desired size
        if subvolume.shape != self.boxsize:
            # Create padded subvolume
            padded = np.zeros(self.boxsize, dtype=subvolume.dtype)
            
            # Calculate padding
            z_pad = half_box[0] - int(coords[0]) if coords[0] < half_box[0] else 0
            y_pad = half_box[1] - int(coords[1]) if coords[1] < half_box[1] else 0
            x_pad = half_box[2] - int(coords[2]) if coords[2] < half_box[2] else 0
            
            # Calculate actual dimensions to copy
            z_size, y_size, x_size = subvolume.shape
            
            # Copy data to padded volume
            padded[z_pad:z_pad+z_size, y_pad:y_pad+y_size, x_pad:x_pad+x_size] = subvolume
            return padded
        
        return subvolume
    
    def get_dataloader(self, batch_size=8, shuffle=True, num_workers=0):
        """
        Get a DataLoader for this dataset.
        
        Args:
            batch_size: batch size for the dataloader
            shuffle: whether to shuffle the dataset
            num_workers: number of worker processes
            
        Returns:
            DataLoader: PyTorch DataLoader
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
