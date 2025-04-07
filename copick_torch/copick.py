import os
import numpy as np
import zarr
import copick
import torch
import pickle
import random
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from collections import Counter
from typing import Dict, List, Tuple, Union, Optional, Any
from torch.utils.data import Dataset, ConcatDataset, Subset
from scipy.ndimage import gaussian_filter

class CopickDataset(Dataset):
    """
    A PyTorch dataset for working with copick data for particle picking tasks.
    
    This implementation focuses on extracting subvolumes around pick coordinates
    with support for data augmentation, caching, and class balancing.
    """

    def __init__(
        self,
        config_path: Union[str, Any] = None,
        copick_root: Optional[Any] = None,
        boxsize: Tuple[int, int, int] = (32, 32, 32),
        augment: bool = False,
        cache_dir: Optional[str] = None,
        cache_format: str = "parquet",  # Can be "pickle" or "parquet"
        seed: Optional[int] = 1717,
        max_samples: Optional[int] = None,
        voxel_spacing: float = 10.0,
        include_background: bool = False,
        background_ratio: float = 0.2,  # Background samples as proportion of particle samples
        min_background_distance: Optional[float] = None,  # Min distance in voxels from particles
        patch_strategy: str = "centered",  # Can be "centered", "random", or "jittered"
        augmentations: Optional[List[str]] = None,  # List of augmentation types to apply
        augmentation_prob: float = 0.5,  # Probability of applying each augmentation
        mixup_alpha: Optional[float] = None,  # Alpha parameter for mixup augmentation
        rotate_axes: Tuple[int, int, int] = (1, 1, 1)  # Enable/disable rotation around each axis (x, y, z)
    ):
        # Validate input: either config_path or copick_root must be provided
        if config_path is None and copick_root is None:
            raise ValueError("Either config_path or copick_root must be provided")
            
        self.config_path = config_path
        self.copick_root = copick_root
        self.boxsize = boxsize
        self.augment = augment
        self.cache_dir = cache_dir
        self.cache_format = cache_format.lower()
        self.seed = seed
        self.max_samples = max_samples
        self.voxel_spacing = voxel_spacing
        self.include_background = include_background
        self.background_ratio = background_ratio
        self.min_background_distance = min_background_distance or max(boxsize)
        self.patch_strategy = patch_strategy
        
        # Augmentation settings
        self.augmentation_prob = augmentation_prob
        self.mixup_alpha = mixup_alpha
        self.rotate_axes = rotate_axes
        
        # Default augmentations if not specified
        self.default_augmentations = ["brightness", "blur", "intensity", "flip", "rotate"]
        self.augmentations = augmentations or self.default_augmentations
        
        # Special augmentations that need additional handling
        self.special_augmentations = ["mixup", "rotate_z"]
        
        # Validate augmentations
        valid_augmentations = self.default_augmentations + self.special_augmentations
        for aug in self.augmentations:
            if aug not in valid_augmentations:
                raise ValueError(f"Unknown augmentation type: {aug}. Valid options are: {valid_augmentations}")
        
        # Validate parameters
        if self.cache_format not in ["pickle", "parquet"]:
            raise ValueError("cache_format must be either 'pickle' or 'parquet'")
            
        if self.patch_strategy not in ["centered", "random", "jittered"]:
            raise ValueError("patch_strategy must be one of 'centered', 'random', or 'jittered'")
        
        # Initialize dataset
        self._set_random_seed()
        self._subvolumes = []
        self._molecule_ids = []
        self._keys = []
        self._is_background = []
        self._load_or_process_data()
        self._compute_sample_weights()

    def _set_random_seed(self):
        """Set random seeds for reproducibility."""
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)

    def _compute_sample_weights(self):
        """Compute sample weights based on class frequency for balancing."""
        # Include special handling for background class if it exists
        class_counts = Counter(self._molecule_ids)
        total_samples = len(self._molecule_ids)
        
        # Assign weights inversely proportional to class frequency
        class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
        
        # Compute weights for each sample
        self.sample_weights = [class_weights[mol_id] for mol_id in self._molecule_ids]

    def _get_cache_path(self):
        """Get the appropriate cache file path based on format."""
        # If we have a copick_root but no config_path, use a hash of the object id
        cache_key = self.config_path
        if cache_key is None and self.copick_root is not None:
            cache_key = f"copick_root_{hash(id(self.copick_root))}"
            
        if self.cache_format == "pickle":
            return os.path.join(
                self.cache_dir,
                f"{cache_key}_{self.boxsize[0]}x{self.boxsize[1]}x{self.boxsize[2]}"
                f"{'_with_bg' if self.include_background else ''}.pkl"
            )
        else:  # parquet
            return os.path.join(
                self.cache_dir,
                f"{cache_key}_{self.boxsize[0]}x{self.boxsize[1]}x{self.boxsize[2]}"
                f"{'_with_bg' if self.include_background else ''}.parquet"
            )

    def _load_or_process_data(self):
        """Load data from cache or process it directly."""
        # If cache_dir is None, process data directly without caching
        if self.cache_dir is None:
            print("Cache directory not specified. Processing data without caching...")
            self._load_data()
            return
            
        # If cache_dir is specified, use caching logic
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = self._get_cache_path()
        
        if os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            
            if self.cache_format == "pickle":
                self._load_from_pickle(cache_file)
            else:  # parquet
                self._load_from_parquet(cache_file)
                
            # Apply max_samples limit if specified
            if self.max_samples is not None and len(self._subvolumes) > self.max_samples:
                indices = np.random.choice(
                    len(self._subvolumes), 
                    self.max_samples, 
                    replace=False
                )
                self._subvolumes = np.array(self._subvolumes)[indices]
                self._molecule_ids = np.array(self._molecule_ids)[indices]
                if self._is_background:
                    self._is_background = np.array(self._is_background)[indices]
        else:
            print("Processing data and creating cache...")
            self._load_data()
            
            # Only save to cache if we actually loaded some data
            if len(self._subvolumes) > 0:
                if self.cache_format == "pickle":
                    self._save_to_pickle(cache_file)
                else:  # parquet
                    self._save_to_parquet(cache_file)
                print(f"Cached data saved to {cache_file}")
            else:
                print("No data loaded, skipping cache creation")

    def _load_from_pickle(self, cache_file):
        """Load dataset from pickle cache."""
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
            self._subvolumes = cached_data.get('subvolumes', [])
            self._molecule_ids = cached_data.get('molecule_ids', [])
            self._keys = cached_data.get('keys', [])
            self._is_background = cached_data.get('is_background', [])
            
            # Handle case where background flag wasn't saved
            if not self._is_background and self.include_background:
                # Initialize all as non-background
                self._is_background = [False] * len(self._subvolumes)

    def _save_to_pickle(self, cache_file):
        """Save dataset to pickle cache."""
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'subvolumes': self._subvolumes,
                'molecule_ids': self._molecule_ids,
                'keys': self._keys,
                'is_background': self._is_background
            }, f)

    def _load_from_parquet(self, cache_file):
        """Load dataset from parquet cache."""
        try:
            df = pd.read_parquet(cache_file)
            
            # Process subvolumes from bytes back to numpy arrays
            self._subvolumes = []
            for idx, row in df.iterrows():
                if isinstance(row['subvolume'], bytes):
                    # Reconstruct numpy array from bytes
                    subvol = np.frombuffer(row['subvolume'], dtype=np.float32)
                    shape = row['shape']
                    if isinstance(shape, list):
                        shape = tuple(shape)
                    subvol = subvol.reshape(shape)
                    self._subvolumes.append(subvol)
                else:
                    raise ValueError(f"Invalid subvolume format: {type(row['subvolume'])}")
            
            # Convert to numpy array
            self._subvolumes = np.array(self._subvolumes)
            
            # Extract other fields
            self._molecule_ids = df['molecule_id'].tolist()
            self._keys = df['key'].tolist() if 'key' in df.columns else []
            
            # Load or initialize background flags
            if 'is_background' in df.columns:
                self._is_background = df['is_background'].tolist()
            else:
                self._is_background = [False] * len(self._subvolumes)
                
            # Reconstruct unique keys if not available
            if not self._keys:
                unique_ids = set()
                for mol_id in self._molecule_ids:
                    if mol_id != -1 and not df.loc[df['molecule_id'] == mol_id, 'is_background'].any():
                        unique_ids.add(mol_id)
                self._keys = sorted(list(unique_ids))
        
        except Exception as e:
            print(f"Error loading from parquet: {str(e)}")
            raise

    def _save_to_parquet(self, cache_file):
        """Save dataset to parquet cache."""
        try:
            # Check if we have any data to save
            if len(self._subvolumes) == 0:
                print("No data to save to parquet")
                return
                
            # Prepare records
            records = []
            for i, (subvol, mol_id, is_bg) in enumerate(zip(self._subvolumes, self._molecule_ids, self._is_background)):
                record = {
                    'subvolume': subvol.tobytes(),
                    'shape': list(subvol.shape),
                    'molecule_id': mol_id,
                    'is_background': is_bg
                }
                records.append(record)
            
            # Add keys information
            key_mapping = []
            for i, key in enumerate(self._keys):
                key_mapping.append({'key_index': i, 'key': key})
            
            # Create and save main dataframe
            df = pd.DataFrame(records)
            
            # Add keys as a column for each row
            df['key'] = df['molecule_id'].apply(
                lambda x: self._keys[x] if x != -1 and x < len(self._keys) else "background"
            )
            
            df.to_parquet(cache_file, index=False)
            
            # Save additional metadata
            metadata_file = cache_file.replace('.parquet', '_metadata.parquet')
            metadata = {
                'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_samples': len(records),
                'unique_molecules': len(self._keys),
                'boxsize': self.boxsize,
                'include_background': self.include_background,
                'background_samples': sum(self._is_background)
            }
            pd.DataFrame([metadata]).to_parquet(metadata_file, index=False)
            
        except Exception as e:
            print(f"Error saving to parquet: {str(e)}")
            raise

    def _extract_subvolume_with_validation(self, tomogram_array, x, y, z):
        """Extract a subvolume with validation checks, applying the selected patch strategy."""
        half_box = np.array(self.boxsize) // 2
        
        # Apply patch strategy
        if self.patch_strategy == "centered":
            # Standard centered extraction
            offset_x, offset_y, offset_z = 0, 0, 0
        elif self.patch_strategy == "random":
            # Random offsets within half_box/4
            max_offset = [size // 4 for size in half_box]
            offset_x = np.random.randint(-max_offset[0], max_offset[0] + 1)
            offset_y = np.random.randint(-max_offset[1], max_offset[1] + 1)
            offset_z = np.random.randint(-max_offset[2], max_offset[2] + 1)
        elif self.patch_strategy == "jittered":
            # Small random jitter for data augmentation
            offset_x = np.random.randint(-2, 3)
            offset_y = np.random.randint(-2, 3)
            offset_z = np.random.randint(-2, 3)
        
        # Apply offsets to coordinates
        x_adj = x + offset_x
        y_adj = y + offset_y
        z_adj = z + offset_z
        
        # Calculate slice indices
        x_start = max(0, int(x_adj - half_box[0]))
        x_end = min(tomogram_array.shape[2], int(x_adj + half_box[0]))
        y_start = max(0, int(y_adj - half_box[1]))
        y_end = min(tomogram_array.shape[1], int(y_adj + half_box[1]))
        z_start = max(0, int(z_adj - half_box[2]))
        z_end = min(tomogram_array.shape[0], int(z_adj + half_box[2]))
        
        # Validate slice ranges
        if x_end <= x_start or y_end <= y_start or z_end <= z_start:
            return None, False, "Invalid slice range"
        
        # Extract subvolume
        subvolume = tomogram_array[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Check if extracted shape matches requested size
        if subvolume.shape != self.boxsize:
            # Need to pad the subvolume
            padded = np.zeros(self.boxsize, dtype=subvolume.dtype)
            
            # Calculate padding amounts
            z_pad_start = max(0, half_box[2] - int(z))
            y_pad_start = max(0, half_box[1] - int(y))
            x_pad_start = max(0, half_box[0] - int(x))
            
            # Calculate end indices
            z_pad_end = min(z_pad_start + (z_end - z_start), self.boxsize[0])
            y_pad_end = min(y_pad_start + (y_end - y_start), self.boxsize[1])
            x_pad_end = min(x_pad_start + (x_end - x_start), self.boxsize[2])
            
            # Copy data
            padded[z_pad_start:z_pad_end, y_pad_start:y_pad_end, x_pad_start:x_pad_end] = subvolume
            return padded, True, "padded"
            
        return subvolume, True, "valid"

    def _load_data(self):
        """Load particle picks data from copick project."""
        # Determine which root to use
        if self.copick_root is not None:
            root = self.copick_root
            print(f"Using provided copick root object")
        else:
            try:
                root = copick.from_file(self.config_path)
                print(f"Loading data from {self.config_path}")
            except Exception as e:
                print(f"Failed to load copick root: {str(e)}")
                return

        # Store all particle coordinates for background sampling
        all_particle_coords = []
        
        for run in root.runs:
            print(f"Processing run: {run.name}")
            
            # Try to load tomogram
            try:
                voxel_spacing_obj = run.get_voxel_spacing(self.voxel_spacing)
                if voxel_spacing_obj is None or not hasattr(voxel_spacing_obj, 'tomograms') or not voxel_spacing_obj.tomograms:
                    print(f"No tomograms found for run {run.name} at voxel spacing {self.voxel_spacing}")
                    continue
                    
                tomogram = voxel_spacing_obj.tomograms[0]
                tomogram_array = tomogram.numpy()
            except Exception as e:
                print(f"Error loading tomogram for run {run.name}: {str(e)}")
                continue

            # Process picks
            run_particle_coords = []  # Store coordinates for this run
            
            for picks in run.get_picks():
                if not picks.from_tool:
                    continue
                    
                object_name = picks.pickable_object_name
                
                try:
                    points, _ = picks.numpy()
                    points = points / self.voxel_spacing
                    
                    for point in points:
                        try:
                            x, y, z = point
                            
                            # Save for background sampling
                            run_particle_coords.append((x, y, z))
                            
                            # Extract subvolume
                            subvolume, is_valid, _ = self._extract_subvolume_with_validation(
                                tomogram_array, x, y, z
                            )
                            
                            if is_valid:
                                self._subvolumes.append(subvolume)
                                
                                if object_name not in self._keys:
                                    self._keys.append(object_name)
                                
                                self._molecule_ids.append(self._keys.index(object_name))
                                self._is_background.append(False)
                        except Exception as e:
                            print(f"Error extracting subvolume: {str(e)}")
                except Exception as e:
                    print(f"Error processing picks for {object_name}: {str(e)}")
            
            # Sample background points for this run if needed
            if self.include_background and run_particle_coords:
                all_particle_coords.extend(run_particle_coords)
                self._sample_background_points(tomogram_array, run_particle_coords)
        
        self._subvolumes = np.array(self._subvolumes)
        self._molecule_ids = np.array(self._molecule_ids)
        self._is_background = np.array(self._is_background)

        # Apply max_samples limit if specified
        if self.max_samples is not None and len(self._subvolumes) > self.max_samples:
            indices = np.random.choice(len(self._subvolumes), self.max_samples, replace=False)
            self._subvolumes = self._subvolumes[indices]
            self._molecule_ids = self._molecule_ids[indices]
            self._is_background = self._is_background[indices]
        
        print(f"Loaded {len(self._subvolumes)} subvolumes with {len(self._keys)} classes")
        print(f"Background samples: {sum(self._is_background)}")

    def _sample_background_points(self, tomogram_array, particle_coords):
        """Sample background points away from particles."""
        if not particle_coords:
            return
            
        # Convert to numpy array for distance calculations
        particle_coords = np.array(particle_coords)
        
        # Calculate number of background samples based on ratio
        num_particles = len(particle_coords)
        num_background = int(num_particles * self.background_ratio)
        
        # Limit attempts to avoid infinite loop
        max_attempts = num_background * 10
        attempts = 0
        bg_points_found = 0
        
        half_box = np.array(self.boxsize) // 2
        
        while bg_points_found < num_background and attempts < max_attempts:
            # Generate random point within tomogram bounds with margin for box extraction
            random_point = np.array([
                np.random.randint(half_box[0], tomogram_array.shape[2] - half_box[0]),
                np.random.randint(half_box[1], tomogram_array.shape[1] - half_box[1]),
                np.random.randint(half_box[2], tomogram_array.shape[0] - half_box[2])
            ])
            
            # Calculate distances to all particles
            distances = np.linalg.norm(particle_coords - random_point, axis=1)
            
            # Check if point is far enough from all particles
            if np.min(distances) >= self.min_background_distance:
                # Extract subvolume
                x, y, z = random_point
                subvolume, is_valid, _ = self._extract_subvolume_with_validation(
                    tomogram_array, x, y, z
                )
                
                if is_valid:
                    self._subvolumes.append(subvolume)
                    self._molecule_ids.append(-1)  # Use -1 to indicate background
                    self._is_background.append(True)
                    bg_points_found += 1
            
            attempts += 1
        
        print(f"Added {bg_points_found} background points after {attempts} attempts")

    def _augment_subvolume(self, subvolume, idx=None):
        """Apply data augmentation to a subvolume.
        
        Args:
            subvolume: The 3D volume to augment
            idx: Optional index for mixup augmentation
            
        Returns:
            Augmented subvolume
        """
        # Apply standard augmentations with probability
        for aug in self.augmentations:
            if random.random() < self.augmentation_prob:
                if aug == "brightness":
                    subvolume = self._brightness(subvolume)
                elif aug == "blur":
                    subvolume = self._gaussian_blur(subvolume)
                elif aug == "intensity":
                    subvolume = self._intensity_scaling(subvolume)
                elif aug == "flip":
                    subvolume = self._flip(subvolume)
                elif aug == "rotate":
                    subvolume = self._rotate(subvolume)
                elif aug == "rotate_z" and "rotate_z" in self.augmentations:
                    subvolume = self._rotate_z(subvolume)
                    
        # Apply mixup if enabled and we have an index to work with
        if "mixup" in self.augmentations and self.mixup_alpha is not None and idx is not None:
            subvolume = self._apply_mixup(subvolume, idx)
            
        return subvolume

    def _brightness(self, volume, max_delta=0.5):
        """Adjust brightness of a volume."""
        delta = np.random.uniform(-max_delta, max_delta)
        return volume + delta

    def _gaussian_blur(self, volume, sigma_range=(0.5, 1.5)):
        """Apply Gaussian blur to a volume."""
        sigma = np.random.uniform(*sigma_range)
        return gaussian_filter(volume, sigma=sigma)

    def _intensity_scaling(self, volume, intensity_range=(0.5, 1.5)):
        """Scale the intensity of a volume."""
        intensity_factor = np.random.uniform(*intensity_range)
        return volume * intensity_factor

    def _flip(self, volume):
        """Randomly flip the volume along one axis."""
        axis = random.randint(0, 2)
        return np.flip(volume, axis=axis)
        
    def _rotate(self, volume):
        """Rotate the volume 90, 180, or 270 degrees around allowed axes."""
        # Filter available axes based on rotate_axes setting
        available_axes = [i for i, allowed in enumerate(self.rotate_axes) if allowed]
        
        if len(available_axes) < 2:
            # Need at least 2 axes for rotation, if not enough are enabled,
            # default to standard x-y rotation
            axis1, axis2 = 0, 1
        else:
            # Select two random axes from the available ones
            axis1, axis2 = random.sample(available_axes, 2)
        
        k = random.randint(1, 3)  # 90, 180, or 270 degrees
        return np.rot90(volume, k=k, axes=(axis1, axis2))
        
    def _rotate_z(self, volume):
        """Apply rotation specifically around z-axis.
        
        This augmentation is useful for tomography data where the z-axis
        often has special significance.
        """
        # For z-rotation, we'll rotate around the first dimension (z-axis)
        # using alternative methods that allow arbitrary angles
        angle = np.random.uniform(0, 360)  # Random angle in degrees
        
        # Get center coordinates
        center_z, center_y, center_x = np.array(volume.shape) // 2
        
        # Create rotation matrix for z-axis rotation
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
        
        # Create coordinates grid
        z, y, x = np.meshgrid(np.arange(volume.shape[0]),
                             np.arange(volume.shape[1]),
                             np.arange(volume.shape[2]),
                             indexing='ij')
        
        # Adjust coordinates to be relative to center
        z -= center_z
        y -= center_y
        x -= center_x
        
        # Stack coordinates and reshape
        coords = np.stack([z.flatten(), y.flatten(), x.flatten()])
        
        # Apply rotation
        rotated_coords = np.dot(rotation_matrix, coords)
        
        # Reshape back and adjust to original coordinate system
        z_rot = rotated_coords[0].reshape(volume.shape) + center_z
        y_rot = rotated_coords[1].reshape(volume.shape) + center_y
        x_rot = rotated_coords[2].reshape(volume.shape) + center_x
        
        # Interpolate using scipy map_coordinates
        from scipy.ndimage import map_coordinates
        rotated_volume = map_coordinates(volume, [z_rot, y_rot, x_rot], order=1, mode='constant')
        
        return rotated_volume
        
    def _apply_mixup(self, subvolume, idx):
        """Apply mixup augmentation by blending with another random sample.
        
        Mixup is a data augmentation technique that creates virtual training examples
        by mixing pairs of inputs and their labels with random proportions.
        
        Args:
            subvolume: The current subvolume being processed
            idx: Index of the current subvolume to avoid mixing with itself
            
        Returns:
            Mixed subvolume
        """
        if len(self._subvolumes) <= 1:
            return subvolume
            
        # Select a different index at random
        other_idx = idx
        while other_idx == idx:
            other_idx = random.randint(0, len(self._subvolumes) - 1)
            
        # Get the other subvolume
        other_subvolume = self._subvolumes[other_idx].copy()
        
        # Sample lambda from beta distribution
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 0.5  # Equal mix if alpha not provided
            
        # Mix the subvolumes
        mixed_subvolume = lam * subvolume + (1 - lam) * other_subvolume
        
        # Note: In a real training scenario, you would also mix the labels
        # For simplicity in the dataset, we're just mixing the inputs
        # The actual label mixing would be done in the training loop
        
        return mixed_subvolume

    def __len__(self):
        """Get the total number of items in the dataset."""
        return len(self._subvolumes)

    def __getitem__(self, idx):
        """Get an item from the dataset."""
        subvolume = self._subvolumes[idx].copy()
        molecule_idx = self._molecule_ids[idx]

        if self.augment:
            subvolume = self._augment_subvolume(subvolume, idx)

        # Normalize
        subvolume = (subvolume - np.mean(subvolume)) / (np.std(subvolume) + 1e-6)
        
        # Add channel dimension and convert to tensor
        subvolume = torch.as_tensor(subvolume[None, ...], dtype=torch.float32)
        
        # For mixup, we could also return the mixing lambda and second class
        # but for simplicity, we just return the standard format for now
        return subvolume, torch.tensor(molecule_idx)

    def get_sample_weights(self):
        """Return sample weights for use in a WeightedRandomSampler."""
        return self.sample_weights

    def keys(self):
        """Get pickable object keys."""
        return self._keys

    def examples(self):
        """Get example volumes for each class."""
        # Check if dataset is empty
        if len(self._subvolumes) == 0 or len(self._molecule_ids) == 0:
            return None, []
            
        class_examples = {}
        example_tensors = []
        example_labels = []
        
        # Get examples for regular classes
        for cls in range(len(self._keys)):
            # Find first index for this class
            for i, mol_id in enumerate(self._molecule_ids):
                if mol_id == cls and cls not in class_examples and \
                   (not self._is_background or not self._is_background[i]):
                    try:
                        volume, _ = self[i]
                        example_tensors.append(volume)
                        example_labels.append(cls)
                        class_examples[cls] = i
                        break
                    except Exception as e:
                        print(f"Error extracting example for class {cls}: {str(e)}")
                        continue
        
        # Add background example if present
        if self.include_background and self._is_background and any(self._is_background):
            # Find first background sample
            for i, is_bg in enumerate(self._is_background):
                if is_bg:
                    try:
                        volume, _ = self[i]
                        example_tensors.append(volume)
                        example_labels.append(-1)  # Use -1 for background
                        break
                    except Exception as e:
                        print(f"Error extracting background example: {str(e)}")
                        continue
        
        if example_tensors:
            return torch.stack(example_tensors), [
                "background" if label == -1 else self._keys[label] 
                for label in example_labels
            ]
        return None, []

    def get_class_distribution(self):
        """Get distribution of classes in the dataset."""
        class_counts = Counter(self._molecule_ids)
        
        # Create a readable distribution
        distribution = {}
        
        # Count background samples if any
        if -1 in class_counts:
            distribution["background"] = class_counts[-1]
            del class_counts[-1]
        
        # Count regular classes
        for cls_idx, count in class_counts.items():
            if 0 <= cls_idx < len(self._keys):
                distribution[self._keys[cls_idx]] = count
        
        return distribution
        
    def stratified_split(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=None):
        """Split the dataset into train, validation, and test sets while preserving class distributions.
        
        Args:
            train_ratio: Proportion of data to use for training
            val_ratio: Proportion of data to use for validation
            test_ratio: Proportion of data to use for testing
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset) as Subset objects
        """
        # Validate ratios
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Ratios must sum to 1.0")
            
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Get indices for each class, including background if present
        class_indices = {}
        for i, mol_id in enumerate(self._molecule_ids):
            if mol_id not in class_indices:
                class_indices[mol_id] = []
            class_indices[mol_id].append(i)
            
        # Shuffle indices for each class
        for mol_id in class_indices:
            np.random.shuffle(class_indices[mol_id])
            
        # Split indices for each class according to ratios
        train_indices = []
        val_indices = []
        test_indices = []
        
        for mol_id, indices in class_indices.items():
            n_samples = len(indices)
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)
            
            # Assign indices to splits
            train_indices.extend(indices[:n_train])
            val_indices.extend(indices[n_train:n_train + n_val])
            test_indices.extend(indices[n_train + n_val:])
            
        # Shuffle the final indices
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)
        
        # Create subset datasets
        train_dataset = Subset(self, train_indices)
        val_dataset = Subset(self, val_indices)
        test_dataset = Subset(self, test_indices)
        
        # Print split information
        print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test samples")
        
        return train_dataset, val_dataset, test_dataset
        
    def balance_classes(self, method='oversample', target_ratio=1.0, exclude_background=False):
        """Balance class distribution in the dataset.
        
        Args:
            method: Balancing method to use ('oversample' or 'undersample')
            target_ratio: For partial balancing (1.0 = perfect balance)
            exclude_background: Whether to exclude background class from balancing
            
        Returns:
            A new CopickDataset instance with balanced classes
        """
        # Validate parameters
        if method not in ['oversample', 'undersample']:
            raise ValueError("method must be either 'oversample' or 'undersample'")
            
        if target_ratio <= 0 or target_ratio > 1.0:
            raise ValueError("target_ratio must be between 0 and 1.0")
            
        # Get class distribution
        class_indices = {}
        for i, mol_id in enumerate(self._molecule_ids):
            # Skip background class if requested
            if exclude_background and mol_id == -1:
                continue
                
            if mol_id not in class_indices:
                class_indices[mol_id] = []
            class_indices[mol_id].append(i)
            
        class_counts = {mol_id: len(indices) for mol_id, indices in class_indices.items()}
        print("Original class distribution:")
        for mol_id, count in class_counts.items():
            class_name = "background" if mol_id == -1 else self._keys[mol_id]
            print(f"  {class_name}: {count} samples")
            
        # Determine target counts
        if method == 'oversample':
            # Oversample minority classes to match majority class
            max_count = max(class_counts.values())
            target_counts = {}
            for mol_id, count in class_counts.items():
                # Calculate the target count for this class
                # At target_ratio=1.0, all classes will have max_count samples
                # At lower ratios, there will be partial balancing
                deficit = max_count - count
                target_counts[mol_id] = count + int(deficit * target_ratio)
                
        else:  # undersample
            # Undersample majority classes to match minority class
            min_count = min(class_counts.values())
            target_counts = {}
            for mol_id, count in class_counts.items():
                # Calculate the target count for this class
                # At target_ratio=1.0, all classes will have min_count samples
                # At lower ratios, there will be partial balancing towards min_count
                excess = count - min_count
                target_counts[mol_id] = count - int(excess * target_ratio)
                
        # Create new balanced dataset
        new_subvolumes = []
        new_molecule_ids = []
        new_is_background = []
        
        # Process each class
        for mol_id, indices in class_indices.items():
            current_count = len(indices)
            target_count = target_counts[mol_id]
            
            if target_count <= current_count:
                # Undersample: randomly select subset of samples
                selected_indices = np.random.choice(indices, target_count, replace=False)
                for idx in selected_indices:
                    new_subvolumes.append(self._subvolumes[idx].copy())
                    new_molecule_ids.append(self._molecule_ids[idx])
                    new_is_background.append(self._is_background[idx])
            else:
                # Oversample: use all original samples and add duplicates with augmentation
                # First, add all original samples
                for idx in indices:
                    new_subvolumes.append(self._subvolumes[idx].copy())
                    new_molecule_ids.append(self._molecule_ids[idx])
                    new_is_background.append(self._is_background[idx])
                    
                # Then add duplicates with augmentation to reach target count
                n_duplicates = target_count - current_count
                duplicate_indices = np.random.choice(indices, n_duplicates, replace=True)
                
                for idx in duplicate_indices:
                    # Apply some basic augmentation to avoid exact duplicates
                    augmented = self._subvolumes[idx].copy()
                    # Apply random augmentations regardless of self.augment setting
                    # This is to ensure variety in the duplicated samples
                    augmented = self._flip(augmented)
                    if random.random() < 0.5:
                        augmented = self._brightness(augmented)
                    if random.random() < 0.5:
                        augmented = self._intensity_scaling(augmented)
                    
                    new_subvolumes.append(augmented)
                    new_molecule_ids.append(self._molecule_ids[idx])
                    new_is_background.append(self._is_background[idx])
        
        # Convert to numpy arrays
        new_subvolumes = np.array(new_subvolumes)
        new_molecule_ids = np.array(new_molecule_ids)
        new_is_background = np.array(new_is_background)
        
        # Create a new dataset with balanced classes
        balanced_dataset = CopickDataset(
            config_path=self.config_path,
            boxsize=self.boxsize,
            augment=self.augment,
            cache_dir=None,  # Don't use caching for the balanced dataset
            seed=self.seed,
            voxel_spacing=self.voxel_spacing,
            include_background=self.include_background,
            patch_strategy=self.patch_strategy
        )
        
        # Replace data with balanced data
        balanced_dataset._subvolumes = new_subvolumes
        balanced_dataset._molecule_ids = new_molecule_ids
        balanced_dataset._is_background = new_is_background
        balanced_dataset._keys = self._keys.copy()
        
        # Compute new sample weights
        balanced_dataset._compute_sample_weights()
        
        # Print final distribution
        balanced_dist = balanced_dataset.get_class_distribution()
        print("Balanced class distribution:")
        for class_name, count in balanced_dist.items():
            print(f"  {class_name}: {count} samples")
            
        return balanced_dataset
    
    def extract_grid_patches(self, patch_size, overlap=0.25, normalize=True, run_index=0, tomo_type='raw'):
        """Extract a grid of patches from a tomogram.
        
        Args:
            patch_size: Int or tuple (z, y, x) for patch dimensions
            overlap: Overlap ratio between adjacent patches (0-1)
            normalize: Whether to normalize patches
            run_index: Index of the run to extract from
            tomo_type: Type of tomogram to extract from ('raw' or 'filtered')
            
        Returns:
            List of extracted patches and their coordinates (z, y, x)
        """
        # Validate parameters
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        elif len(patch_size) != 3:
            raise ValueError("patch_size must be an integer or tuple of 3 integers")
            
        if overlap < 0 or overlap >= 1:
            raise ValueError("overlap must be between 0 and 1")
            
        # Get tomogram data
        try:
            root = copick.from_file(self.config_path)
            if not root.runs:
                raise ValueError("No runs found in the copick project")
                
            # Use the specified run
            if run_index >= len(root.runs):
                raise ValueError(f"Run index {run_index} out of range. Only {len(root.runs)} runs available.")
                
            run = root.runs[run_index]
            
            # Get the tomogram based on voxel spacing
            tomogram = run.get_voxel_spacing(self.voxel_spacing).tomograms[0]
            
            # Get the appropriate tomogram type
            if tomo_type == 'raw':
                tomogram_array = tomogram.numpy()
            elif tomo_type == 'filtered':
                # Check if filtered data is available
                if hasattr(tomogram, 'filtered') and tomogram.filtered is not None:
                    tomogram_array = tomogram.filtered.numpy()
                else:
                    print("Warning: Filtered tomogram not available, using raw tomogram instead")
                    tomogram_array = tomogram.numpy()
            else:
                raise ValueError(f"Invalid tomogram type: {tomo_type}. Must be 'raw' or 'filtered'")
            
            # Calculate stride (step size between patches)
            stride_z = int(patch_size[0] * (1 - overlap))
            stride_y = int(patch_size[1] * (1 - overlap))
            stride_x = int(patch_size[2] * (1 - overlap))
            
            # Ensure stride is at least 1
            stride_z = max(1, stride_z)
            stride_y = max(1, stride_y)
            stride_x = max(1, stride_x)
            
            # Calculate number of patches in each dimension
            n_patches_z = 1 + (tomogram_array.shape[0] - patch_size[0]) // stride_z
            n_patches_y = 1 + (tomogram_array.shape[1] - patch_size[1]) // stride_y
            n_patches_x = 1 + (tomogram_array.shape[2] - patch_size[2]) // stride_x
            
            # Initialize results
            patches = []
            coordinates = []
            
            # Extract patches
            for iz in range(n_patches_z):
                z_start = iz * stride_z
                z_end = z_start + patch_size[0]
                if z_end > tomogram_array.shape[0]:
                    continue
                    
                for iy in range(n_patches_y):
                    y_start = iy * stride_y
                    y_end = y_start + patch_size[1]
                    if y_end > tomogram_array.shape[1]:
                        continue
                        
                    for ix in range(n_patches_x):
                        x_start = ix * stride_x
                        x_end = x_start + patch_size[2]
                        if x_end > tomogram_array.shape[2]:
                            continue
                            
                        # Extract the patch
                        patch = tomogram_array[z_start:z_end, y_start:y_end, x_start:x_end].copy()
                        
                        # Normalize if requested
                        if normalize:
                            # Center and scale to unit variance
                            patch = (patch - np.mean(patch)) / (np.std(patch) + 1e-6)
                            
                        # Record patch and its center coordinates
                        patches.append(patch)
                        coordinates.append((z_start + patch_size[0]//2,
                                         y_start + patch_size[1]//2,
                                         x_start + patch_size[2]//2))
            
            print(f"Extracted {len(patches)} patches of size {patch_size} with {overlap:.2f} overlap")
            return patches, coordinates
            
        except Exception as e:
            print(f"Error extracting grid patches: {str(e)}")
            raise
            
    def extract_from_region(self, x_range, y_range, z_range, tomo_type='raw'):
        """Extract a specific region from a tomogram.
        
        Args:
            x_range: Tuple of (min_x, max_x) in voxel space
            y_range: Tuple of (min_y, max_y) in voxel space
            z_range: Tuple of (min_z, max_z) in voxel space
            tomo_type: Type of tomogram to extract from ('raw' or 'filtered')
            
        Returns:
            A numpy array containing the extracted region
        """
        # Validate ranges
        if not all(isinstance(r, tuple) and len(r) == 2 for r in [x_range, y_range, z_range]):
            raise ValueError("Range parameters must be tuples of (min, max)")
            
        # Get tomogram data
        try:
            root = copick.from_file(self.config_path)
            if not root.runs:
                raise ValueError("No runs found in the copick project")
                
            # Use the first run by default
            run = root.runs[0]
            
            # Get the tomogram based on voxel spacing
            tomogram = run.get_voxel_spacing(self.voxel_spacing).tomograms[0]
            
            # Get the appropriate tomogram type
            if tomo_type == 'raw':
                tomogram_array = tomogram.numpy()
            elif tomo_type == 'filtered':
                # Check if filtered data is available
                if hasattr(tomogram, 'filtered') and tomogram.filtered is not None:
                    tomogram_array = tomogram.filtered.numpy()
                else:
                    print("Warning: Filtered tomogram not available, using raw tomogram instead")
                    tomogram_array = tomogram.numpy()
            else:
                raise ValueError(f"Invalid tomogram type: {tomo_type}. Must be 'raw' or 'filtered'")
                
            # Extract the requested region
            min_x, max_x = x_range
            min_y, max_y = y_range
            min_z, max_z = z_range
            
            # Convert to integer indices and ensure they're within bounds
            min_z = max(0, int(min_z))
            max_z = min(tomogram_array.shape[0], int(max_z))
            min_y = max(0, int(min_y))
            max_y = min(tomogram_array.shape[1], int(max_y))
            min_x = max(0, int(min_x))
            max_x = min(tomogram_array.shape[2], int(max_x))
            
            # Extract the region
            region = tomogram_array[min_z:max_z, min_y:max_y, min_x:max_x]
            
            if region.size == 0:
                raise ValueError("Extracted region is empty. Check range parameters.")
                
            return region
            
        except Exception as e:
            print(f"Error extracting region from tomogram: {str(e)}")
            raise

    def extract_grid_patches(self, patch_size, overlap=0.25, normalize=True, run_index=0, tomo_type='raw'):
        """Extract a grid of patches from a tomogram.
        
        Args:
            patch_size: Int or tuple (z, y, x) for patch dimensions
            overlap: Overlap ratio between adjacent patches (0-1)
            normalize: Whether to normalize patches
            run_index: Index of the run to extract from
            tomo_type: Type of tomogram to extract from ('raw' or 'filtered')
            
        Returns:
            List of extracted patches and their coordinates (z, y, x)
        """
        # Validate parameters
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        elif len(patch_size) != 3:
            raise ValueError("patch_size must be an integer or tuple of 3 integers")
            
        if overlap < 0 or overlap >= 1:
            raise ValueError("overlap must be between 0 and 1")
            
        # Get tomogram data
        try:
            root = copick.from_file(self.config_path)
            if not root.runs:
                raise ValueError("No runs found in the copick project")
                
            # Use the specified run
            if run_index >= len(root.runs):
                raise ValueError(f"Run index {run_index} out of range. Only {len(root.runs)} runs available.")
                
            run = root.runs[run_index]
            
            # Get the tomogram based on voxel spacing
            tomogram = run.get_voxel_spacing(self.voxel_spacing).tomograms[0]
            
            # Get the appropriate tomogram type
            if tomo_type == 'raw':
                tomogram_array = tomogram.numpy()
            elif tomo_type == 'filtered':
                # Check if filtered data is available
                if hasattr(tomogram, 'filtered') and tomogram.filtered is not None:
                    tomogram_array = tomogram.filtered.numpy()
                else:
                    print("Warning: Filtered tomogram not available, using raw tomogram instead")
                    tomogram_array = tomogram.numpy()
            else:
                raise ValueError(f"Invalid tomogram type: {tomo_type}. Must be 'raw' or 'filtered'")
            
            # Calculate stride (step size between patches)
            stride_z = int(patch_size[0] * (1 - overlap))
            stride_y = int(patch_size[1] * (1 - overlap))
            stride_x = int(patch_size[2] * (1 - overlap))
            
            # Ensure stride is at least 1
            stride_z = max(1, stride_z)
            stride_y = max(1, stride_y)
            stride_x = max(1, stride_x)
            
            # Calculate number of patches in each dimension
            n_patches_z = 1 + (tomogram_array.shape[0] - patch_size[0]) // stride_z
            n_patches_y = 1 + (tomogram_array.shape[1] - patch_size[1]) // stride_y
            n_patches_x = 1 + (tomogram_array.shape[2] - patch_size[2]) // stride_x
            
            # Initialize results
            patches = []
            coordinates = []
            
            # Extract patches
            for iz in range(n_patches_z):
                z_start = iz * stride_z
                z_end = z_start + patch_size[0]
                if z_end > tomogram_array.shape[0]:
                    continue
                    
                for iy in range(n_patches_y):
                    y_start = iy * stride_y
                    y_end = y_start + patch_size[1]
                    if y_end > tomogram_array.shape[1]:
                        continue
                        
                    for ix in range(n_patches_x):
                        x_start = ix * stride_x
                        x_end = x_start + patch_size[2]
                        if x_end > tomogram_array.shape[2]:
                            continue
                            
                        # Extract the patch
                        patch = tomogram_array[z_start:z_end, y_start:y_end, x_start:x_end].copy()
                        
                        # Normalize if requested
                        if normalize:
                            # Center and scale to unit variance
                            patch = (patch - np.mean(patch)) / (np.std(patch) + 1e-6)
                            
                        # Record patch and its center coordinates
                        patches.append(patch)
                        coordinates.append((z_start + patch_size[0]//2,
                                         y_start + patch_size[1]//2,
                                         x_start + patch_size[2]//2))
            
            print(f"Extracted {len(patches)} patches of size {patch_size} with {overlap:.2f} overlap")
            return patches, coordinates
            
        except Exception as e:
            print(f"Error extracting grid patches: {str(e)}")
            raise
            
    def extract_from_region(self, x_range, y_range, z_range, tomo_type='raw'):
        """Extract a specific region from a tomogram.
        
        Args:
            x_range: Tuple of (min_x, max_x) in voxel space
            y_range: Tuple of (min_y, max_y) in voxel space
            z_range: Tuple of (min_z, max_z) in voxel space
            tomo_type: Type of tomogram to extract from ('raw' or 'filtered')
            
        Returns:
            A numpy array containing the extracted region
        """
        # Validate ranges
        if not all(isinstance(r, tuple) and len(r) == 2 for r in [x_range, y_range, z_range]):
            raise ValueError("Range parameters must be tuples of (min, max)")
            
        # Get tomogram data
        try:
            root = copick.from_file(self.config_path)
            if not root.runs:
                raise ValueError("No runs found in the copick project")
                
            # Use the first run by default
            run = root.runs[0]
            
            # Get the tomogram based on voxel spacing
            tomogram = run.get_voxel_spacing(self.voxel_spacing).tomograms[0]
            
            # Get the appropriate tomogram type
            if tomo_type == 'raw':
                tomogram_array = tomogram.numpy()
            elif tomo_type == 'filtered':
                # Check if filtered data is available
                if hasattr(tomogram, 'filtered') and tomogram.filtered is not None:
                    tomogram_array = tomogram.filtered.numpy()
                else:
                    print("Warning: Filtered tomogram not available, using raw tomogram instead")
                    tomogram_array = tomogram.numpy()
            else:
                raise ValueError(f"Invalid tomogram type: {tomo_type}. Must be 'raw' or 'filtered'")
                
            # Extract the requested region
            min_x, max_x = x_range
            min_y, max_y = y_range
            min_z, max_z = z_range
            
            # Convert to integer indices and ensure they're within bounds
            min_z = max(0, int(min_z))
            max_z = min(tomogram_array.shape[0], int(max_z))
            min_y = max(0, int(min_y))
            max_y = min(tomogram_array.shape[1], int(max_y))
            min_x = max(0, int(min_x))
            max_x = min(tomogram_array.shape[2], int(max_x))
            
            # Extract the region
            region = tomogram_array[min_z:max_z, min_y:max_y, min_x:max_x]
            
            if region.size == 0:
                raise ValueError("Extracted region is empty. Check range parameters.")
                
            return region
            
        except Exception as e:
            print(f"Error extracting region from tomogram: {str(e)}")
            raise