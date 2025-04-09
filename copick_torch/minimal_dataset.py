"""
A minimal CopickDataset implementation without caching or augmentation.
"""

import numpy as np
import torch
import copick
import zarr
from torch.utils.data import Dataset
from collections import Counter
import logging

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
        dataset_id,
        overlay_root="/tmp/test/",
        boxsize=(48, 48, 48),
        voxel_spacing=10.012,
        include_background=False,
        background_ratio=0.2,
        min_background_distance=None,
        max_samples=None
    ):
        """
        Initialize a MinimalCopickDataset.
        
        Args:
            dataset_id: Dataset ID from the CZ cryoET Data Portal
            overlay_root: Root directory for the overlay storage
            boxsize: Size of the subvolumes to extract (z, y, x)
            voxel_spacing: Voxel spacing to use for extraction
            include_background: Whether to include background samples
            background_ratio: Ratio of background to particle samples
            min_background_distance: Minimum distance from particles for background samples
            max_samples: Maximum number of samples to use (None for no limit)
        """
        self.dataset_id = dataset_id
        self.overlay_root = overlay_root
        self.boxsize = boxsize
        self.voxel_spacing = voxel_spacing
        self.include_background = include_background
        self.background_ratio = background_ratio
        self.min_background_distance = min_background_distance or max(boxsize)
        self.max_samples = max_samples
        
        # Initialize data structures
        self._points = []  # List of (x, y, z) coordinates
        self._labels = []  # List of class indices
        self._class_names = []  # List of class names
        self._is_background = []  # List of booleans indicating if a sample is background
        self._tomogram_zarr = None  # Zarr array for the tomogram
        
        # Load the data
        self._load_data()
        
    def _load_data(self):
        """Load data from the copick project."""
        try:
            # Create copick root object
            self.copick_root = copick.from_czcdp_datasets([self.dataset_id], overlay_root=self.overlay_root)
            logger.info(f"Created copick root from dataset ID: {self.dataset_id}")
            
            # Get pickable objects
            self._object_names = [obj.name for obj in self.copick_root.pickable_objects]
            logger.info(f"Found pickable objects: {self._object_names}")
            
            # Process each run
            all_points = []
            all_labels = []
            all_is_background = []
            
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
                    self._tomogram_zarr = zarr.open(tomogram.zarr())["0"]
                    logger.info(f"Loaded tomogram with shape {self._tomogram_zarr.shape}")
                    
                    # Store all particle coordinates for background sampling
                    all_particle_coords = []
                    
                    # Process picks for each object type
                    for picks in run.get_picks():
                        if not picks.from_tool:
                            continue
                            
                        object_name = picks.pickable_object_name
                        
                        if object_name not in self._object_names:
                            logger.warning(f"Object {object_name} not in pickable objects, adding it")
                            self._object_names.append(object_name)
                        
                        class_idx = self._object_names.index(object_name)
                        
                        try:
                            points, _ = picks.numpy()
                            if len(points) == 0:
                                logger.warning(f"No points found for {object_name}")
                                continue
                                
                            logger.info(f"Found {len(points)} points for {object_name}")
                            
                            # Store the points and labels
                            for point in points:
                                all_points.append(point)
                                all_labels.append(class_idx)
                                all_is_background.append(False)
                                all_particle_coords.append(point)
                        except Exception as e:
                            logger.error(f"Error processing picks for {object_name}: {e}")
                    
                    # Sample background points if requested
                    if self.include_background and all_particle_coords:
                        num_particles = len(all_particle_coords)
                        num_background = int(num_particles * self.background_ratio)
                        
                        logger.info(f"Sampling {num_background} background points")
                        
                        bg_points = self._sample_background_points(
                            self._tomogram_zarr.shape,
                            all_particle_coords,
                            num_background,
                            self.min_background_distance
                        )
                        
                        for point in bg_points:
                            all_points.append(point)
                            all_labels.append(-1)  # -1 indicates background
                            all_is_background.append(True)
                            
                except Exception as e:
                    logger.error(f"Error processing tomogram for run {run.name}: {e}")
                    continue
            
            # Store the processed data
            self._points = all_points
            self._labels = all_labels
            self._is_background = all_is_background
            
            # Apply max_samples limit if specified
            if self.max_samples is not None and len(self._points) > self.max_samples:
                indices = np.random.choice(len(self._points), self.max_samples, replace=False)
                self._points = [self._points[i] for i in indices]
                self._labels = [self._labels[i] for i in indices]
                self._is_background = [self._is_background[i] for i in indices]
            
            logger.info(f"Dataset loaded with {len(self._points)} samples")
            
            # Print class distribution
            self._print_class_distribution()
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
            
    def _print_class_distribution(self):
        """Print the distribution of classes in the dataset."""
        class_counts = Counter(self._labels)
        
        # Create a readable distribution
        distribution = {}
        
        # Count background samples if any
        if -1 in class_counts:
            distribution["background"] = class_counts[-1]
            del class_counts[-1]
        
        # Count regular classes
        for cls_idx, count in class_counts.items():
            if 0 <= cls_idx < len(self._object_names):
                distribution[self._object_names[cls_idx]] = count
        
        logger.info("Class distribution:")
        for class_name, count in distribution.items():
            logger.info(f"  {class_name}: {count} samples")
            
        return distribution
            
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
        
    def extract_subvolume(self, point):
        """
        Extract a cubic subvolume centered around a point.
        
        Args:
            point: (x, y, z) coordinates
            
        Returns:
            Extracted subvolume as a numpy array
        """
        # Check if tomogram is loaded
        if self._tomogram_zarr is None:
            raise ValueError("No tomogram loaded")
            
        # Get dimensions of the tomogram
        z_dim, y_dim, x_dim = self._tomogram_zarr.shape
        
        # Convert coordinates to indices
        x_idx = int(point[0] / self.voxel_spacing)
        y_idx = int(point[1] / self.voxel_spacing)
        z_idx = int(point[2] / self.voxel_spacing)
        
        # Calculate half box sizes
        half_x = self.boxsize[2] // 2
        half_y = self.boxsize[1] // 2
        half_z = self.boxsize[0] // 2
        
        # Calculate bounds with boundary checking
        x_start = max(0, x_idx - half_x)
        x_end = min(x_dim, x_idx + half_x)
        y_start = max(0, y_idx - half_y)
        y_end = min(y_dim, y_idx + half_y)
        z_start = max(0, z_idx - half_z)
        z_end = min(z_dim, z_idx + half_z)
        
        # Extract subvolume
        subvolume = self._tomogram_zarr[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Pad if necessary
        if subvolume.shape != self.boxsize:
            padded = np.zeros(self.boxsize, dtype=subvolume.dtype)
            
            # Calculate padding dimensions
            pad_z = min(z_end - z_start, self.boxsize[0])
            pad_y = min(y_end - y_start, self.boxsize[1])
            pad_x = min(x_end - x_start, self.boxsize[2])
            
            # Calculate padding offsets (center the subvolume in the padded volume)
            z_offset = (self.boxsize[0] - pad_z) // 2
            y_offset = (self.boxsize[1] - pad_y) // 2
            x_offset = (self.boxsize[2] - pad_x) // 2
            
            # Insert subvolume into padded volume
            padded[
                z_offset:z_offset+pad_z,
                y_offset:y_offset+pad_y,
                x_offset:x_offset+pad_x
            ] = subvolume
            
            return padded
            
        return subvolume
        
    def __len__(self):
        """Get the length of the dataset."""
        return len(self._points)
        
    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (subvolume, label)
        """
        # Get the point and label
        point = self._points[idx]
        label = self._labels[idx]
        
        # Extract the subvolume
        subvolume = self.extract_subvolume(point)
        
        # Normalize
        if np.std(subvolume) > 0:
            subvolume = (subvolume - np.mean(subvolume)) / np.std(subvolume)
            
        # Add channel dimension and convert to tensor
        subvolume_tensor = torch.as_tensor(subvolume[None, ...], dtype=torch.float32)
        
        return subvolume_tensor, label
        
    def keys(self):
        """Get the list of class names."""
        # Add background class if included
        if self.include_background:
            return self._object_names + ["background"]
        return self._object_names
        
    def get_class_distribution(self):
        """Get the distribution of classes in the dataset."""
        distribution = Counter()
        
        for label in self._labels:
            if label == -1:
                distribution["background"] += 1
            else:
                distribution[self._object_names[label]] += 1
                
        return dict(distribution)
        
    def get_sample_weights(self):
        """
        Compute sample weights for balanced sampling.
        
        Returns:
            List of weights for each sample
        """
        # Count instances of each class
        class_counts = Counter(self._labels)
        total_samples = len(self._labels)
        
        # Compute inverse frequency weights
        weights = []
        for label in self._labels:
            weight = total_samples / class_counts[label]
            weights.append(weight)
            
        return weights
