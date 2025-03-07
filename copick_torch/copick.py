import os
import numpy as np
import zarr
import copick
import torch
import pickle
import random
from collections import Counter
from typing import Dict, List, Tuple, Union, Optional
from torch.utils.data import Dataset, ConcatDataset
from scipy.ndimage import gaussian_filter

class CopickDataset(Dataset):
    """
    A PyTorch dataset for working with copick data for particle picking tasks.
    
    This implementation focuses on extracting subvolumes around pick coordinates
    with support for data augmentation, caching, and class balancing.
    """

    def __init__(
        self,
        config_path: str,
        boxsize: Tuple[int, int, int] = (32, 32, 32),
        augment: bool = False,
        cache_dir: Optional[str] = None,
        seed: Optional[int] = 1717,
        max_samples: Optional[int] = None,
        voxel_spacing: float = 10.0
    ):
        self.config_path = config_path
        self.boxsize = boxsize
        self.augment = augment
        self.cache_dir = cache_dir
        self.seed = seed
        self.max_samples = max_samples
        self.voxel_spacing = voxel_spacing
        
        # Initialize dataset
        self._set_random_seed()
        self._subvolumes = []
        self._molecule_ids = []
        self._keys = []
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
        class_counts = Counter(self._molecule_ids)
        total_samples = len(self._molecule_ids)
        class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
        self.sample_weights = [class_weights[mol_id] for mol_id in self._molecule_ids]

    def _load_or_process_data(self):
        """Load data from cache or process it directly."""
        # If cache_dir is None, process data directly without caching
        if self.cache_dir is None:
            print("Cache directory not specified. Processing data without caching...")
            self._load_data()
            return
            
        # If cache_dir is specified, use caching logic
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(
            self.cache_dir, 
            f"copick_cache_{self.boxsize[0]}x{self.boxsize[1]}x{self.boxsize[2]}.pkl"
        )
        
        if os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self._subvolumes = cached_data.get('subvolumes', [])
                self._molecule_ids = cached_data.get('molecule_ids', [])
                self._keys = cached_data.get('keys', [])
                
                # Apply max_samples limit if specified
                if self.max_samples is not None and len(self._subvolumes) > self.max_samples:
                    indices = np.random.choice(
                        len(self._subvolumes), 
                        self.max_samples, 
                        replace=False
                    )
                    self._subvolumes = np.array(self._subvolumes)[indices]
                    self._molecule_ids = np.array(self._molecule_ids)[indices]
        else:
            print("Processing data and creating cache...")
            self._load_data()
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'subvolumes': self._subvolumes,
                    'molecule_ids': self._molecule_ids,
                    'keys': self._keys
                }, f)
            print(f"Cached data saved to {cache_file}")

    def _load_data(self):
        """Load particle picks data from copick project."""
        print(f"Loading data from {self.config_path}")
        
        # Load copick root
        try:
            root = copick.from_file(self.config_path)
        except Exception as e:
            print(f"Failed to load copick root: {str(e)}")
            return

        for run in root.runs:
            print(f"Processing run: {run.name}")
            
            # Try to load tomogram
            try:
                tomogram = run.get_voxel_spacing(self.voxel_spacing).tomograms[0]
                tomogram_array = tomogram.numpy()
            except Exception as e:
                print(f"Error loading tomogram for run {run.name}: {str(e)}")
                continue

            # Process picks
            for picks in run.picks:
                if not picks.from_tool:
                    continue
                    
                object_name = picks.pickable_object_name
                
                try:
                    points, _ = picks.numpy()
                    points = points / self.voxel_spacing
                    
                    for point in points:
                        try:
                            x, y, z = point
                            subvolume = self._extract_subvolume(tomogram_array, x, y, z)
                            self._subvolumes.append(subvolume)
                            
                            if object_name not in self._keys:
                                self._keys.append(object_name)
                            
                            self._molecule_ids.append(self._keys.index(object_name))
                        except ValueError:
                            pass
                except Exception as e:
                    print(f"Error processing picks for {object_name}: {str(e)}")
        
        self._subvolumes = np.array(self._subvolumes)
        self._molecule_ids = np.array(self._molecule_ids)

        # Apply max_samples limit if specified
        if self.max_samples is not None and len(self._subvolumes) > self.max_samples:
            indices = np.random.choice(len(self._subvolumes), self.max_samples, replace=False)
            self._subvolumes = self._subvolumes[indices]
            self._molecule_ids = self._molecule_ids[indices]
        
        print(f"Loaded {len(self._subvolumes)} subvolumes with {len(self._keys)} classes")

    def _extract_subvolume(self, tomogram_array, x, y, z):
        """Extract a subvolume centered at the given coordinates."""
        half_box = np.array(self.boxsize) // 2
        
        # Calculate slice indices
        x_start = max(0, int(x - half_box[0]))
        x_end = min(tomogram_array.shape[2], int(x + half_box[0]))
        y_start = max(0, int(y - half_box[1]))
        y_end = min(tomogram_array.shape[1], int(y + half_box[1]))
        z_start = max(0, int(z - half_box[2]))
        z_end = min(tomogram_array.shape[0], int(z + half_box[2]))
        
        # Extract subvolume
        subvolume = tomogram_array[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Pad if necessary
        if subvolume.shape != self.boxsize:
            padded = np.zeros(self.boxsize, dtype=subvolume.dtype)
            
            # Calculate padding amounts
            z_pad_start = max(0, half_box[2] - int(z))
            y_pad_start = max(0, half_box[1] - int(y))
            x_pad_start = max(0, half_box[0] - int(x))
            
            z_subvol_start = max(0, int(z) - half_box[2])
            y_subvol_start = max(0, int(y) - half_box[1])
            x_subvol_start = max(0, int(x) - half_box[0])
            
            # Calculate end indices
            z_pad_end = min(z_pad_start + subvolume.shape[0], self.boxsize[0])
            y_pad_end = min(y_pad_start + subvolume.shape[1], self.boxsize[1])
            x_pad_end = min(x_pad_start + subvolume.shape[2], self.boxsize[2])
            
            z_subvol_end = min(z_subvol_start + (z_pad_end - z_pad_start), subvolume.shape[0])
            y_subvol_end = min(y_subvol_start + (y_pad_end - y_pad_start), subvolume.shape[1])
            x_subvol_end = min(x_subvol_start + (x_pad_end - x_pad_start), subvolume.shape[2])
            
            # Copy data
            padded[z_pad_start:z_pad_end, y_pad_start:y_pad_end, x_pad_start:x_pad_end] = subvolume[
                z_subvol_start:z_subvol_end, 
                y_subvol_start:y_subvol_end, 
                x_subvol_start:x_subvol_end
            ]
            
            return padded
            
        return subvolume

    def _augment_subvolume(self, subvolume):
        """Apply data augmentation to a subvolume."""
        if random.random() < 0.5:
            subvolume = self._brightness(subvolume)
        if random.random() < 0.5:
            subvolume = self._gaussian_blur(subvolume)
        if random.random() < 0.5:
            subvolume = self._intensity_scaling(subvolume)
        if random.random() < 0.5:
            subvolume = self._flip(subvolume)
        if random.random() < 0.5:
            subvolume = self._rotate(subvolume)
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
        """Rotate the volume 90, 180, or 270 degrees around a random axis."""
        axis1, axis2 = random.sample(range(3), 2)
        k = random.randint(1, 3)  # 90, 180, or 270 degrees
        return np.rot90(volume, k=k, axes=(axis1, axis2))

    def __len__(self):
        """Get the total number of items in the dataset."""
        return len(self._subvolumes)

    def __getitem__(self, idx):
        """Get an item from the dataset."""
        subvolume = self._subvolumes[idx].copy()
        molecule_idx = self._molecule_ids[idx]

        if self.augment:
            subvolume = self._augment_subvolume(subvolume)

        # Normalize
        subvolume = (subvolume - np.mean(subvolume)) / (np.std(subvolume) + 1e-6)
        
        # Add channel dimension and convert to tensor
        subvolume = torch.as_tensor(subvolume[None, ...], dtype=torch.float32)
        return subvolume, torch.tensor(molecule_idx)

    def get_sample_weights(self):
        """Return sample weights for use in a WeightedRandomSampler."""
        return self.sample_weights

    def keys(self):
        """Get pickable object keys."""
        return self._keys

    def examples(self):
        """Get example volumes for each class."""
        class_examples = {}
        example_tensors = []
        example_labels = []
        
        for cls in range(len(self._keys)):
            # Find first index for this class
            for i, mol_id in enumerate(self._molecule_ids):
                if mol_id == cls and cls not in class_examples:
                    volume, _ = self[i]
                    example_tensors.append(volume)
                    example_labels.append(cls)
                    class_examples[cls] = i
                    break
        
        if example_tensors:
            return torch.stack(example_tensors), [self._keys[i] for i in example_labels]
        return None, []