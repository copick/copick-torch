"""
A minimal CopickDataset implementation with support for saving/loading and multi-tomogram processing.
"""

import numpy as np
import torch
import copick
import zarr
import os
import json
from torch.utils.data import Dataset
from collections import Counter
import logging
from types import SimpleNamespace
from tqdm import tqdm

logger = logging.getLogger(__name__)

class MinimalCopickDataset(Dataset):
    """
    A minimal PyTorch dataset for working with copick data that returns (image, label) pairs.
    
    Unlike the SimpleCopickDataset, this implementation:
    1. Does not use caching (loads data on-the-fly)
    2. Does not include augmentation
    3. Has minimal dependencies
    4. Focuses on correct label mapping
    
    This is useful for generating documentation examples and testing.
    """
    
    def __init__(
        self,
        dataset_id=None,
        overlay_root="/tmp/test/",
        boxsize=(48, 48, 48),
        voxel_spacing=10.012,
        include_background=False,
        background_ratio=0.2,
        min_background_distance=None,
        max_samples=None,
        preload=False,
        tomogram_data=None,
        name_to_label=None
    ):
        """
        Initialize a MinimalCopickDataset.
        
        Args:
            dataset_id: Dataset ID from the CZ cryoET Data Portal (optional if tomogram_data is provided)
            overlay_root: Root directory for the overlay storage
            boxsize: Size of the subvolumes to extract (z, y, x)
            voxel_spacing: Voxel spacing to use for extraction
            include_background: Whether to include background samples
            background_ratio: Ratio of background to particle samples
            min_background_distance: Minimum distance from particles for background samples
            max_samples: Maximum number of samples to use (None for no limit)
            preload: Whether to preload all subvolumes into memory
            tomogram_data: Optional list of (tomogram_zarr, picks) tuples for direct initialization
            name_to_label: Optional dictionary mapping pickable object names to class IDs
        """
        self.dataset_id = dataset_id
        self.overlay_root = overlay_root
        self.cube_size = boxsize[0] if isinstance(boxsize, tuple) else boxsize
        self.boxsize = (self.cube_size, self.cube_size, self.cube_size) if isinstance(boxsize, int) else boxsize
        self.voxel_spacing = voxel_spacing
        self.include_background = include_background
        self.background_ratio = background_ratio
        self.min_background_distance = min_background_distance or max(self.boxsize)
        self.max_samples = max_samples
        self.preload = preload
        
        # Initialize data structures
        self._points = []  # List of (x, y, z) coordinates or SimpleNamespace objects
        self._labels = []  # List of class indices
        self._class_names = []  # List of class names
        self._is_background = []  # List of booleans indicating if a sample is background
        self._tomogram_data = tomogram_data or []  # List of (tomogram_zarr, picks) tuples
        self._samples = []  # List of (class_id, coordinates, point, tomo_idx) tuples
        self.subvolumes = None  # Storage for preloaded subvolumes
        
        # Setup name to label mapping
        self.name_to_label = name_to_label or {}
        
        # Load the data if dataset_id or tomogram_data is provided
        if dataset_id is not None or tomogram_data is not None:
            self._load_data()
        
    def _load_data(self):
        """Load data from the copick project or provided tomogram data."""
        try:
            # If tomogram_data is provided, use it directly
            if self._tomogram_data:
                logger.info(f"Using provided tomogram data with {len(self._tomogram_data)} tomograms")
                self._process_tomogram_data()
                return
                
            # Otherwise, load from copick project
            # Create copick root object
            self.copick_root = copick.from_czcdp_datasets([self.dataset_id], overlay_root=self.overlay_root)
            logger.info(f"Created copick root from dataset ID: {self.dataset_id}")
            
            # Get pickable objects and create name_to_label mapping if not provided
            if not self.name_to_label:
                self.name_to_label = {obj.name: idx for idx, obj in enumerate(self.copick_root.pickable_objects)}
                
            self._object_names = list(self.name_to_label.keys())
            logger.info(f"Found pickable objects: {self._object_names}")
            
            # Process each run
            for run in self.copick_root.runs:
                logger.info(f"Processing run: {run.name}")
                
                # Get tomogram
                try:
                    voxel_spacing_obj = run.get_voxel_spacing(self.voxel_spacing)
                    if not voxel_spacing_obj or not voxel_spacing_obj.tomograms:
                        logger.warning(f"No tomograms found for run {run.name} at voxel spacing {self.voxel_spacing}")
                        continue
                        
                    # Find a denoised tomogram if available, otherwise use the first one
                    tomogram = [t for t in voxel_spacing_obj.tomograms if 'wbp-denoised' in t.tomo_type]
                    if not tomogram:
                        tomogram = voxel_spacing_obj.tomograms[0]
                    else:
                        tomogram = tomogram[0]
                        
                    # Open zarr array
                    tomogram_zarr = zarr.open(tomogram.zarr())["0"]
                    logger.info(f"Loaded tomogram with shape {tomogram_zarr.shape}")
                    
                    # Get picks for the run
                    picks = run.picks
                    if not picks:
                        logger.warning(f"No picks found for run {run.name}")
                        continue
                        
                    # Add to tomogram data
                    self._tomogram_data.append((tomogram_zarr, picks))
                    
                except Exception as e:
                    logger.error(f"Error processing tomogram for run {run.name}: {e}")
                    continue
            
            # Process the tomogram data
            self._process_tomogram_data()
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _process_tomogram_data(self):
        """Process tomogram data to extract samples."""
        # Reset samples list
        self._samples = []
        
        # Process each tomogram and its picks
        for tomo_idx, (tomogram_zarr, picks) in enumerate(self._tomogram_data):
            logger.info(f"Processing tomogram {tomo_idx+1}/{len(self._tomogram_data)}")
            
            # Process each set of picks
            for pick in picks:
                class_name = pick.pickable_object_name
                
                # Update name_to_label if needed
                if class_name not in self.name_to_label:
                    logger.warning(f"Object {class_name} not in name_to_label mapping, adding it")
                    self.name_to_label[class_name] = len(self.name_to_label)
                
                class_id = self.name_to_label[class_name]
                
                # Add each point to samples
                for point in pick.points:
                    # Extract coordinates
                    if hasattr(point, 'location'):
                        # Handle SimpleNamespace object
                        coord = (point.location.z, point.location.y, point.location.x)
                    else:
                        # Handle array-like object
                        coord = point
                    
                    # Store in samples list
                    self._samples.append((class_id, coord, point, tomo_idx))
        
        logger.info(f"Processed {len(self._samples)} samples from {len(self._tomogram_data)} tomograms")
        
        # Create class names list if empty
        if not hasattr(self, '_object_names') or not self._object_names:
            self._object_names = list(self.name_to_label.keys())
        
        # Apply max_samples limit if specified
        if self.max_samples is not None and len(self._samples) > self.max_samples:
            indices = np.random.choice(len(self._samples), self.max_samples, replace=False)
            self._samples = [self._samples[i] for i in indices]
            logger.info(f"Applied max_samples limit, reduced to {len(self._samples)} samples")
        
        # Preload all subvolumes if requested
        if self.preload:
            logger.info(f"Preloading {len(self._samples)} subvolumes...")
            self.subvolumes = []
            for idx in tqdm(range(len(self._samples))):
                class_id, _, point, tomo_idx = self._samples[idx]
                tomogram_zarr = self._tomogram_data[tomo_idx][0]
                subvolume = self._extract_subvolume(point, tomogram_zarr)
                self.subvolumes.append((class_id, subvolume))
            logger.info(f"Preloaded {len(self.subvolumes)} subvolumes")
        
        # Print class distribution
        self._print_class_distribution()
    
    def _print_class_distribution(self):
        """Print the distribution of classes in the dataset."""
        class_counts = Counter()
        
        for class_id, _, _, _ in self._samples:
            if class_id == -1:
                class_counts["background"] += 1
            else:
                class_name = list(self.name_to_label.keys())[list(self.name_to_label.values()).index(class_id)]
                class_counts[class_name] += 1
        
        logger.info("Class distribution:")
        for class_name, count in class_counts.items():
            logger.info(f"  {class_name}: {count} samples")
            
        return class_counts
    
    def _extract_subvolume(self, point, tomogram_zarr):
        """
        Extract a cubic subvolume centered around a particle pick point.
        
        Args:
            point: Point object or coordinate tuple (x, y, z)
            tomogram_zarr: Zarr array for the tomogram
            
        Returns:
            Extracted subvolume as a numpy array
        """
        # Get dimensions of the tomogram
        z_dim, y_dim, x_dim = tomogram_zarr.shape
        
        # Extract coordinates from the pick point and convert to indices
        if hasattr(point, 'location'):
            # Handle SimpleNamespace object
            x_idx = int(point.location.x / self.voxel_spacing)
            y_idx = int(point.location.y / self.voxel_spacing)
            z_idx = int(point.location.z / self.voxel_spacing)
        else:
            # Handle array-like object
            x_idx = int(point[0] / self.voxel_spacing)
            y_idx = int(point[1] / self.voxel_spacing)
            z_idx = int(point[2] / self.voxel_spacing)
        
        # Calculate subvolume bounds with boundary checking
        half_size = self.cube_size // 2
        
        z_start = max(0, z_idx - half_size)
        z_end = min(z_dim, z_idx + half_size)
        
        y_start = max(0, y_idx - half_size)
        y_end = min(y_dim, y_idx + half_size)
        
        x_start = max(0, x_idx - half_size)
        x_end = min(x_dim, x_idx + half_size)
        
        # Extract the subvolume
        subvolume = tomogram_zarr[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Ensure the subvolume has the expected size by padding if necessary
        final_shape = (self.cube_size, self.cube_size, self.cube_size)
        if subvolume.shape != final_shape:
            padded = np.zeros(final_shape, dtype=subvolume.dtype)
            # Calculate the slice to copy the data into
            z_slice = slice(0, subvolume.shape[0])
            y_slice = slice(0, subvolume.shape[1])
            x_slice = slice(0, subvolume.shape[2])
            padded[z_slice, y_slice, x_slice] = subvolume
            subvolume = padded
            
        return subvolume
    
    def __len__(self):
        """Get the length of the dataset."""
        return len(self._samples)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (subvolume, label)
        """
        if self.preload and self.subvolumes is not None:
            class_id, subvolume = self.subvolumes[idx]
        else:
            class_id, _, point, tomo_idx = self._samples[idx]
            tomogram_zarr = self._tomogram_data[tomo_idx][0]
            subvolume = self._extract_subvolume(point, tomogram_zarr)
        
        # Normalize
        if np.std(subvolume) > 0:
            subvolume = (subvolume - np.mean(subvolume)) / np.std(subvolume)
            
        # Add channel dimension and convert to tensor
        subvolume_tensor = torch.as_tensor(subvolume[None, ...], dtype=torch.float32)
        
        return subvolume_tensor, class_id
    
    def save(self, save_dir):
        """
        Save the dataset to disk for later reloading
        
        Args:
            save_dir: Directory to save the dataset
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the subvolumes as a torch tensor file
        if self.preload and self.subvolumes is not None:
            # Extract class IDs and subvolumes into separate lists
            class_ids = []
            subvolumes = []
            for class_id, subvolume in self.subvolumes:
                class_ids.append(class_id)
                subvolumes.append(subvolume)
            
            # Convert to tensors and save
            subvolumes_tensor = torch.tensor(np.array(subvolumes))
            class_ids_tensor = torch.tensor(class_ids)
            torch.save(subvolumes_tensor, os.path.join(save_dir, 'subvolumes.pt'))
            torch.save(class_ids_tensor, os.path.join(save_dir, 'class_ids.pt'))
        else:
            # Save sample information for later extraction
            sample_data = []
            for class_id, coord, point, tomo_idx in self._samples:
                # We need to convert the point to a serializable format
                if hasattr(point, 'location'):
                    point_data = {
                        'x': point.location.x,
                        'y': point.location.y,
                        'z': point.location.z,
                        'pickable_object_name': getattr(point, 'pickable_object_name', None)
                    }
                else:
                    point_data = {
                        'x': point[0],
                        'y': point[1],
                        'z': point[2],
                        'pickable_object_name': None
                    }
                
                sample_data.append({
                    'class_id': class_id,
                    'coord': coord,
                    'point': point_data,
                    'tomo_idx': tomo_idx
                })
            
            with open(os.path.join(save_dir, 'samples.json'), 'w') as f:
                json.dump(sample_data, f)
        
        # Save metadata
        metadata = {
            'voxel_spacing': self.voxel_spacing,
            'cube_size': self.cube_size,
            'boxsize': self.boxsize,
            'preload': self.preload,
            'name_to_label': self.name_to_label,
            'include_background': self.include_background,
            'background_ratio': self.background_ratio,
            'min_background_distance': self.min_background_distance
        }
        
        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        # Save tomogram information
        tomogram_info = []
        for idx, (tomogram_zarr, _) in enumerate(self._tomogram_data):
            tomo_data = {
                'index': idx,
                'path': getattr(tomogram_zarr, 'path', str(tomogram_zarr)),
                'shape': list(tomogram_zarr.shape)
            }
            tomogram_info.append(tomo_data)
        
        with open(os.path.join(save_dir, 'tomogram_info.json'), 'w') as f:
            json.dump(tomogram_info, f)
            
        logger.info(f"Dataset saved to {save_dir}")
    
    @classmethod
    def load(cls, save_dir, tomogram_data=None, transform=None):
        """
        Load a previously saved dataset
        
        Args:
            save_dir: Directory where the dataset was saved
            tomogram_data: List of (tomogram_zarr, picks) tuples (if None, will attempt to use saved paths)
            transform: Optional transforms to apply to the subvolumes
        
        Returns:
            Loaded MinimalCopickDataset instance
        """
        # Load metadata
        with open(os.path.join(save_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        # Check if we need to load the tomogram data
        if tomogram_data is None:
            with open(os.path.join(save_dir, 'tomogram_info.json'), 'r') as f:
                tomogram_info = json.load(f)
                logger.info(f"Note: You need to provide the tomogram_data if samples were not preloaded.")
                logger.info(f"Expected tomogram count: {len(tomogram_info)}")
                for tomo in tomogram_info:
                    logger.info(f"  Tomogram {tomo['index']}: shape {tomo['shape']}")
        
        # Create a new dataset instance with minimal initialization
        dataset = cls.__new__(cls)
        dataset.voxel_spacing = metadata['voxel_spacing']
        dataset.cube_size = metadata['cube_size']
        dataset.boxsize = metadata.get('boxsize', (dataset.cube_size, dataset.cube_size, dataset.cube_size))
        dataset.preload = metadata['preload']
        dataset.name_to_label = metadata['name_to_label']
        dataset.include_background = metadata.get('include_background', False)
        dataset.background_ratio = metadata.get('background_ratio', 0.2)
        dataset.min_background_distance = metadata.get('min_background_distance', dataset.cube_size)
        dataset._tomogram_data = tomogram_data or []
        dataset._object_names = list(dataset.name_to_label.keys())
        
        # Load either preloaded subvolumes or sample information
        if os.path.exists(os.path.join(save_dir, 'subvolumes.pt')):
            # Load preloaded subvolumes
            subvolumes_tensor = torch.load(os.path.join(save_dir, 'subvolumes.pt'))
            class_ids_tensor = torch.load(os.path.join(save_dir, 'class_ids.pt'))
            
            dataset.subvolumes = []
            for i in range(len(class_ids_tensor)):
                class_id = class_ids_tensor[i].item()
                subvolume = subvolumes_tensor[i].numpy()
                dataset.subvolumes.append((class_id, subvolume))
            
            # Reconstruct samples list (minimally, just for length information)
            dataset._samples = [(cls_id, None, None, 0) for cls_id, _ in dataset.subvolumes]
            
        elif os.path.exists(os.path.join(save_dir, 'samples.json')):
            # Load sample information
            with open(os.path.join(save_dir, 'samples.json'), 'r') as f:
                sample_data = json.load(f)
            
            # This requires the tomogram data to be provided
            if tomogram_data is None:
                raise ValueError("tomogram_data must be provided when loading a dataset that wasn't preloaded")
            
            # Reconstruct samples
            dataset._samples = []
            for sample in sample_data:
                # Reconstruct point object with location namespace
                point = SimpleNamespace()
                point.location = SimpleNamespace()
                point.location.x = sample['point']['x']
                point.location.y = sample['point']['y']
                point.location.z = sample['point']['z']
                if sample['point']['pickable_object_name'] is not None:
                    point.pickable_object_name = sample['point']['pickable_object_name']
                
                dataset._samples.append((sample['class_id'], sample['coord'], point, sample['tomo_idx']))
            
            dataset.subvolumes = None
        
        logger.info(f"Loaded dataset with {len(dataset._samples)} samples")
        return dataset
    
    def keys(self):
        """Get the list of class names."""
        # Add background class if included
        if self.include_background:
            return list(self.name_to_label.keys()) + ["background"]
        return list(self.name_to_label.keys())
        
    def get_class_distribution(self):
        """Get the distribution of classes in the dataset."""
        distribution = Counter()
        
        for class_id, _, _, _ in self._samples:
            if class_id == -1:
                distribution["background"] += 1
            else:
                class_name = list(self.name_to_label.keys())[list(self.name_to_label.values()).index(class_id)]
                distribution[class_name] += 1
                
        return dict(distribution)
        
    def get_sample_weights(self):
        """
        Compute sample weights for balanced sampling.
        
        Returns:
            List of weights for each sample
        """
        # Count instances of each class
        class_counts = Counter()
        for class_id, _, _, _ in self._samples:
            class_counts[class_id] += 1
            
        total_samples = len(self._samples)
        
        # Compute inverse frequency weights
        weights = []
        for class_id, _, _, _ in self._samples:
            weight = total_samples / class_counts[class_id]
            weights.append(weight)
            
        return weights
        
    def _sample_background_points(self, tomogram_shape, particle_coords, num_points, min_distance):
        """
        Sample random background points away from particles.
        
        Args:
            tomogram_shape: Shape of the tomogram (z, y, x)
            particle_coords: List of particle coordinates
            num_points: Number of background points to sample
            min_distance: Minimum distance from particles
            
        Returns:
            List of background points
        """
        # Convert to numpy array for vectorized calculations
        if particle_coords:
            particle_array = np.array(particle_coords)
        else:
            particle_array = np.array([[0, 0, 0]])  # Dummy point if no particles
            
        # Get dimensions
        z_dim, y_dim, x_dim = tomogram_shape
        half_box = np.array(self.boxsize) // 2
        
        # Define valid ranges
        x_range = (half_box[2], x_dim - half_box[2])
        y_range = (half_box[1], y_dim - half_box[1])
        z_range = (half_box[0], z_dim - half_box[0])
        
        # Sample points
        bg_points = []
        max_attempts = num_points * 10
        attempts = 0
        
        while len(bg_points) < num_points and attempts < max_attempts:
            # Generate random point
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            z = np.random.uniform(z_range[0], z_range[1])
            point = np.array([x, y, z])
            
            # Check distance to all particles
            if particle_array is not None:
                distances = np.linalg.norm(particle_array - point, axis=1)
                min_dist = np.min(distances)
                
                if min_dist >= min_distance:
                    bg_points.append(point)
            else:
                # No particles to avoid
                bg_points.append(point)
                
            attempts += 1
            
        logger.info(f"Sampled {len(bg_points)} background points after {attempts} attempts")
        return bg_points
