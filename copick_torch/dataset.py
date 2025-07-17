import logging
import os
import pickle
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import copick
import numpy as np
import pandas as pd
import torch
import zarr

# Import these at module level to avoid pickling issues
from scipy.ndimage import binary_dilation, gaussian_filter
from skimage import measure
from skimage.transform import resize
from torch.utils.data import ConcatDataset, Dataset, Subset

from .augmentations import FourierAugment3D


class SimpleDatasetMixin:
    """
    A mixin class that modifies datasets to return simple (image, label) pairs.

    This modifies the __getitem__ method to return a tuple of (subvolume, label_index)
    rather than the more complex dictionary format.
    """

    def __getitem__(self, idx):
        """
        Get an item from the dataset, returning a simple (subvolume, label) pair.

        This simplifies the original __getitem__ method to return just an image tensor
        and a class label integer.

        Returns:
            tuple: (subvolume, label)
        """
        # Get the subvolume using the original method
        # The original method may apply augmentations if self.augment is True
        subvolume = self._subvolumes[idx].copy()
        molecule_idx = self._molecule_ids[idx]

        if self.augment:
            # Apply augmentations if enabled
            subvolume = self._augment_subvolume(subvolume, idx)

        # Normalize subvolume
        subvolume = (subvolume - np.mean(subvolume)) / (np.std(subvolume) + 1e-6)

        # Add channel dimension and convert to tensor
        subvolume = torch.as_tensor(subvolume[None, ...], dtype=torch.float32)

        # Return the subvolume and class index as a simple tuple
        return subvolume, molecule_idx


class SimpleCopickDataset(SimpleDatasetMixin, Dataset):
    """
    A simplified PyTorch dataset for working with copick data that returns (image, label) pairs.

    This implementation is a wrapper around the original CopickDataset that modifies the
    __getitem__ method to return a simpler format suitable for standard training pipelines.
    """

    def __init__(
        self,
        config_path: Union[str, Any] = None,
        copick_root: Optional[Any] = None,
        boxsize: Tuple[int, int, int] = (32, 32, 32),
        augment: bool = False,
        cache_dir: Optional[str] = None,
        cache_format: str = "parquet",
        seed: Optional[int] = 1717,
        max_samples: Optional[int] = None,
        voxel_spacing: float = 10.0,
        include_background: bool = False,
        background_ratio: float = 0.2,
        min_background_distance: Optional[float] = None,
        patch_strategy: str = "centered",
        debug_mode: bool = False,
        dataset_id: Optional[int] = None,
        overlay_root: str = "/tmp/test/",
    ):
        """
        Initialize a SimpleCopickDataset.

        Args:
            config_path: Path to the copick config file or CopickConfig object
            copick_root: Copick root object (alternative to config_path)
            boxsize: Size of the subvolumes to extract (z, y, x)
            augment: Whether to apply data augmentation
            cache_dir: Directory to cache extracted subvolumes
            cache_format: Format for caching ('pickle' or 'parquet')
            seed: Random seed for reproducibility
            max_samples: Maximum number of samples to use
            voxel_spacing: Voxel spacing to use for extraction
            include_background: Whether to include background samples
            background_ratio: Ratio of background to particle samples
            min_background_distance: Minimum distance from particles for background samples
            patch_strategy: Strategy for extracting patches ('centered', 'random', or 'jittered')
            debug_mode: Whether to enable debug mode
        """
        # Validate input: either config_path, copick_root, or dataset_id must be provided
        if config_path is None and copick_root is None and dataset_id is None:
            raise ValueError("Either config_path, copick_root, or dataset_id must be provided")

        self.config_path = config_path
        self.copick_root = copick_root
        self.dataset_id = dataset_id
        self.overlay_root = overlay_root

        # If dataset_id is provided but not copick_root, create it here
        if self.dataset_id is not None and self.copick_root is None:
            try:
                import copick

                self.copick_root = copick.from_czcdp_datasets([self.dataset_id], overlay_root=self.overlay_root)
                print(f"Created copick root from dataset ID: {self.dataset_id}")
            except Exception as e:
                print(f"Error creating copick root from dataset ID: {e}")
                raise
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
        self.debug_mode = debug_mode

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
        # If we have a copick_root but no config_path, use dataset IDs
        cache_key = self.config_path
        if cache_key is None and self.copick_root is not None:
            # Try to get dataset IDs from the datasets attribute
            try:
                dataset_ids = []
                for dataset in self.copick_root.datasets:
                    if hasattr(dataset, "id"):
                        dataset_ids.append(str(dataset.id))

                if dataset_ids:
                    # Use the dataset IDs in order as the cache key
                    dataset_ids_str = "_".join(dataset_ids)
                    cache_key = f"datasets_{dataset_ids_str}"
                else:
                    # Fallback if no dataset IDs found
                    cache_key = "copick_root_unknown"
            except (AttributeError, IndexError):
                # Fallback if datasets attribute doesn't exist
                if hasattr(self.copick_root, "dataset_ids"):
                    dataset_ids = [str(did) for did in self.copick_root.dataset_ids]
                    cache_key = f"datasets_{'_'.join(dataset_ids)}"
                else:
                    # Last resort fallback
                    cache_key = f"copick_root_{hash(str(self.copick_root))}"

        if self.cache_format == "pickle":
            return os.path.join(
                self.cache_dir,
                f"{cache_key}_{self.boxsize[0]}x{self.boxsize[1]}x{self.boxsize[2]}"
                f"_{self.voxel_spacing}"
                f"{'_with_bg' if self.include_background else ''}.pkl",
            )
        else:  # parquet
            return os.path.join(
                self.cache_dir,
                f"{cache_key}_{self.boxsize[0]}x{self.boxsize[1]}x{self.boxsize[2]}"
                f"_{self.voxel_spacing}"
                f"{'_with_bg' if self.include_background else ''}.parquet",
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
                indices = np.random.choice(len(self._subvolumes), self.max_samples, replace=False)
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
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)
            self._subvolumes = cached_data.get("subvolumes", [])
            self._molecule_ids = cached_data.get("molecule_ids", [])
            self._keys = cached_data.get("keys", [])
            self._is_background = cached_data.get("is_background", [])

            # Handle case where background flag wasn't saved
            if not self._is_background and self.include_background:
                # Initialize all as non-background
                self._is_background = [False] * len(self._subvolumes)

    def _save_to_pickle(self, cache_file):
        """Save dataset to pickle cache."""
        with open(cache_file, "wb") as f:
            pickle.dump(
                {
                    "subvolumes": self._subvolumes,
                    "molecule_ids": self._molecule_ids,
                    "keys": self._keys,
                    "is_background": self._is_background,
                },
                f,
            )

    def _load_from_parquet(self, cache_file):
        """Load dataset from parquet cache."""
        try:
            df = pd.read_parquet(cache_file)

            # Process subvolumes from bytes back to numpy arrays
            self._subvolumes = []
            for idx, row in df.iterrows():
                if isinstance(row["subvolume"], bytes):
                    # Reconstruct numpy array from bytes
                    subvol = np.frombuffer(row["subvolume"], dtype=np.float32)
                    shape = row["shape"]
                    if isinstance(shape, list):
                        shape = tuple(shape)
                    subvol = subvol.reshape(shape)
                    self._subvolumes.append(subvol)
                else:
                    raise ValueError(f"Invalid subvolume format: {type(row['subvolume'])}")

            # Convert to numpy array
            self._subvolumes = np.array(self._subvolumes)

            # Extract other fields
            self._molecule_ids = df["molecule_id"].tolist()
            self._keys = df["key"].tolist() if "key" in df.columns else []

            # Load or initialize background flags
            if "is_background" in df.columns:
                self._is_background = df["is_background"].tolist()
            else:
                self._is_background = [False] * len(self._subvolumes)

            # Reconstruct unique keys if not available
            if not self._keys:
                unique_ids = set()
                for mol_id in self._molecule_ids:
                    if mol_id != -1 and not df.loc[df["molecule_id"] == mol_id, "is_background"].any():
                        unique_ids.add(mol_id)
                self._keys = sorted(unique_ids)

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
            for subvol, mol_id, is_bg in zip(self._subvolumes, self._molecule_ids, self._is_background):
                record = {
                    "subvolume": subvol.tobytes(),
                    "shape": list(subvol.shape),
                    "molecule_id": mol_id,
                    "is_background": is_bg,
                }
                records.append(record)

            # Add keys information
            key_mapping = []
            for i, key in enumerate(self._keys):
                key_mapping.append({"key_index": i, "key": key})

            # Create and save main dataframe
            df = pd.DataFrame(records)

            # Add keys as a column for each row
            df["key"] = df["molecule_id"].apply(
                lambda x: self._keys[x] if x != -1 and x < len(self._keys) else "background",
            )

            df.to_parquet(cache_file, index=False)

            # Save additional metadata
            metadata_file = cache_file.replace(".parquet", "_metadata.parquet")
            metadata = {
                "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_samples": len(records),
                "unique_molecules": len(self._keys),
                "boxsize": self.boxsize,
                "include_background": self.include_background,
                "background_samples": sum(self._is_background),
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
            print("Using provided copick root object")
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
                if (
                    voxel_spacing_obj is None
                    or not hasattr(voxel_spacing_obj, "tomograms")
                    or not voxel_spacing_obj.tomograms
                ):
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
                            subvolume, is_valid, _ = self._extract_subvolume_with_validation(tomogram_array, x, y, z)

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
            random_point = np.array(
                [
                    np.random.randint(half_box[0], tomogram_array.shape[2] - half_box[0]),
                    np.random.randint(half_box[1], tomogram_array.shape[1] - half_box[1]),
                    np.random.randint(half_box[2], tomogram_array.shape[0] - half_box[2]),
                ],
            )

            # Calculate distances to all particles
            distances = np.linalg.norm(particle_coords - random_point, axis=1)

            # Check if point is far enough from all particles
            if np.min(distances) >= self.min_background_distance:
                # Extract subvolume
                x, y, z = random_point
                subvolume, is_valid, _ = self._extract_subvolume_with_validation(tomogram_array, x, y, z)

                if is_valid:
                    self._subvolumes.append(subvolume)
                    self._molecule_ids.append(-1)  # Use -1 to indicate background
                    self._is_background.append(True)
                    bg_points_found += 1

            attempts += 1

        print(f"Added {bg_points_found} background points after {attempts} attempts")

    def _augment_subvolume(self, subvolume, idx=None):
        """Apply data augmentation to a subvolume.

        This simplified version applies basic augmentations only (no mixup).

        Args:
            subvolume: The 3D volume to augment
            idx: Optional index for mixup augmentation (not used in this version)

        Returns:
            Augmented subvolume
        """
        # Apply random brightness adjustment
        if random.random() < 0.3:
            delta = np.random.uniform(-0.5, 0.5)
            subvolume = subvolume + delta

        # Apply random Gaussian blur
        if random.random() < 0.2:
            sigma = np.random.uniform(0.5, 1.5)
            subvolume = gaussian_filter(subvolume, sigma=sigma)

        # Apply random intensity scaling
        if random.random() < 0.2:
            factor = np.random.uniform(0.5, 1.5)
            subvolume = subvolume * factor

        # Apply random flip
        if random.random() < 0.2:
            axis = random.randint(0, 2)
            subvolume = np.flip(subvolume, axis=axis)

        # Apply random rotation
        if random.random() < 0.2:
            k = random.randint(1, 3)  # 90, 180, or 270 degrees
            axes = tuple(random.sample([0, 1, 2], 2))  # Select 2 random axes
            subvolume = np.rot90(subvolume, k=k, axes=axes)

        # Apply Fourier domain augmentation
        if random.random() < 0.3:  # 30% chance to apply Fourier augmentation
            fourier_aug = FourierAugment3D(freq_mask_prob=0.3, phase_noise_std=0.1, intensity_scaling_range=(0.8, 1.2))
            subvolume = fourier_aug(subvolume)

        return subvolume

    def __len__(self):
        """Get the total number of items in the dataset."""
        return len(self._subvolumes)

    def get_sample_weights(self):
        """Return sample weights for use in a WeightedRandomSampler."""
        return self.sample_weights

    def keys(self):
        """Get pickable object keys."""
        return self._keys

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


class SplicedMixupDataset(SimpleCopickDataset):
    """
    A dataset that loads zarr arrays into memory and performs balanced sampling with mixup splicing.

    This dataset extends SimpleCopickDataset to add experimental-synthetic data splicing capabilities,
    keeping zarr arrays in memory for faster loading and using balanced sampling by default.
    """

    def __init__(
        self,
        exp_dataset_id: int,
        synth_dataset_id: int,
        synth_run_id: str = "16487",
        overlay_root: str = "/tmp/test/",
        boxsize: Tuple[int, int, int] = (48, 48, 48),
        augment: bool = True,
        cache_dir: Optional[str] = None,
        cache_format: str = "parquet",
        seed: Optional[int] = 1717,
        max_samples: Optional[int] = None,
        voxel_spacing: float = 10.0,
        include_background: bool = False,
        background_ratio: float = 0.2,
        min_background_distance: Optional[float] = None,
        blend_sigma: float = 2.0,  # Controls the standard deviation of Gaussian blending at boundaries
        mixup_alpha: float = 0.2,
        debug_mode: bool = False,
    ):
        """
        Initialize the SplicedMixupDataset.

        Args:
            exp_dataset_id: Dataset ID for the experimental dataset
            synth_dataset_id: Dataset ID for the synthetic dataset
            synth_run_id: Run ID for the synthetic dataset (default: "16487")
            overlay_root: Root directory for the overlay storage (default: "/tmp/test/")
            boxsize: Size of the subvolumes to extract (z, y, x)
            augment: Whether to apply data augmentation
            cache_dir: Directory to cache extracted subvolumes
            cache_format: Format for caching ('pickle' or 'parquet')
            seed: Random seed for reproducibility
            max_samples: Maximum number of samples to use
            voxel_spacing: Voxel spacing to use for extraction
            include_background: Whether to include background samples
            background_ratio: Ratio of background to particle samples
            min_background_distance: Minimum distance from particles for background samples
            blend_sigma: Controls the standard deviation of Gaussian blending at boundaries
            mixup_alpha: Alpha parameter for mixup augmentation
            debug_mode: Whether to enable debug mode
        """
        # Save specific parameters
        self.exp_dataset_id = exp_dataset_id
        self.synth_dataset_id = synth_dataset_id
        self.synth_run_id = synth_run_id
        self.overlay_root = overlay_root
        self.blend_sigma = blend_sigma
        self.mixup_alpha = mixup_alpha

        # Initialize load flags and storage for zarr arrays
        self._zarr_loaded = False
        self._exp_zarr_data = None
        self._synth_zarr_data = None
        self._synth_mask_data = {}

        # Load copick roots
        self._load_copick_roots()

        # Initialize the parent class (SimpleCopickDataset)
        # We'll override certain methods to use our in-memory zarr arrays
        super().__init__(
            copick_root=self.exp_root if hasattr(self, "exp_root") else None,  # Use experimental data as the base
            boxsize=boxsize,
            augment=augment,
            cache_dir=cache_dir,
            cache_format=cache_format,
            seed=seed,
            max_samples=max_samples,
            voxel_spacing=voxel_spacing,
            include_background=include_background,
            background_ratio=background_ratio,
            min_background_distance=min_background_distance,
            patch_strategy="centered",  # Always use centered for splicing
            debug_mode=debug_mode,
        )

        # Load zarr arrays into memory if not already loaded
        self._ensure_zarr_loaded()

        # Initialize dataset with a small number of samples to make sure the parent initialization works
        # We will create our own samples from zarr arrays after parent initialization

        # Generate synthetic samples directly from zarr arrays
        self._generate_synthetic_samples()

    def _generate_synthetic_samples(self):
        """Generate synthetic samples directly from zarr arrays."""
        print("Generating synthetic samples from zarr arrays...")
        # Clear any existing samples
        self._subvolumes = []
        self._molecule_ids = []
        self._keys = []
        self._is_background = []

        num_samples = 100  # Default number of samples
        if self.max_samples is not None:
            num_samples = self.max_samples

        # Generate samples (half from experimental data, half from synthetic+experimental splice)
        num_exp_samples = num_samples // 2
        num_synth_samples = num_samples - num_exp_samples

        # Generate experimental samples
        for _ in range(num_exp_samples):
            # Extract a random crop from experimental data
            exp_crop = self._extract_random_crop(self._exp_zarr_data, self.boxsize)
            self._subvolumes.append(exp_crop)
            self._molecule_ids.append(-1)  # Background class
            self._is_background.append(True)

        # Generate synthetic+experimental spliced samples
        for _ in range(num_synth_samples):
            # Get a random mask name
            mask_names = list(self._synth_mask_data.keys())
            mask_name = random.choice(mask_names)

            # Extract a bounding box
            bbox_info = self._extract_bounding_box(self._synth_mask_data[mask_name], mask_name)

            if bbox_info is not None:
                # Extract a random crop from experimental data
                exp_crop = self._extract_random_crop(self._exp_zarr_data, self.boxsize)

                # Splice the volumes
                spliced_volume = self._splice_volumes(bbox_info["synth_region"], bbox_info["region_mask"], exp_crop)

                # Add to dataset
                self._subvolumes.append(spliced_volume)

                # Get or create molecule index
                if bbox_info["object_name"] not in self._keys:
                    self._keys.append(bbox_info["object_name"])
                molecule_idx = self._keys.index(bbox_info["object_name"])

                self._molecule_ids.append(molecule_idx)
                self._is_background.append(False)

        # Convert to numpy arrays
        self._subvolumes = np.array(self._subvolumes)
        self._molecule_ids = np.array(self._molecule_ids)
        self._is_background = np.array(self._is_background)

        # Compute sample weights
        self._compute_sample_weights()

        print(f"Generated {len(self._subvolumes)} samples with {len(self._keys)} classes")
        print(f"Background samples: {sum(self._is_background)}")
        print(f"Class distribution: {self.get_class_distribution()}")

    def _load_copick_roots(self):
        """Load the experimental and synthetic copick roots."""
        try:
            print(f"Loading experimental dataset {self.exp_dataset_id} and synthetic dataset {self.synth_dataset_id}")
            self.exp_root = copick.from_czcdp_datasets([self.exp_dataset_id], overlay_root=self.overlay_root)
            self.synth_root = copick.from_czcdp_datasets([self.synth_dataset_id], overlay_root=self.overlay_root)

            print(f"Experimental dataset: {len(self.exp_root.runs)} runs")
            print(f"Synthetic dataset: {len(self.synth_root.runs)} runs")

            # Filter synthetic dataset to only include the specified run
            if self.synth_run_id:
                print(f"Filtering synthetic dataset to only use run {self.synth_run_id}")
                filtered_runs = [run for run in self.synth_root.runs if run.meta.name == self.synth_run_id]
                if filtered_runs:
                    print(f"Found run {self.synth_run_id}. Using only this run.")
                    # Store the filtered run for use in loading zarr data
                    self.synth_root._filtered_run = filtered_runs[0]
                else:
                    print(f"Run {self.synth_run_id} not found in synthetic dataset. Using all available runs.")
                    self.synth_root._filtered_run = None
        except Exception as e:
            print(f"Error loading CoPick roots: {str(e)}")
            raise

    def _ensure_zarr_loaded(self):
        """Load zarr arrays into memory if not already loaded."""
        if not self._zarr_loaded:
            self._load_experimental_zarr()
            self._load_synthetic_zarr()
            self._load_segmentation_masks()
            self._zarr_loaded = True
            print("Zarr arrays loaded into memory.")

    def _load_experimental_zarr(self):
        """Load experimental tomogram into memory."""
        try:
            # Get available tomograms from experimental dataset
            exp_tomograms = self._get_available_tomograms(self.exp_root, self.voxel_spacing)

            if not exp_tomograms:
                raise ValueError(f"No experimental tomograms found with voxel spacing {self.voxel_spacing}")

            # Select the first tomogram
            exp_tomogram_obj = exp_tomograms[0]
            exp_zarr = zarr.open(exp_tomogram_obj.zarr(), "r")
            self._exp_zarr_data = exp_zarr["0"][:]

            # Normalize tomogram data
            self._exp_zarr_data = (self._exp_zarr_data - np.mean(self._exp_zarr_data)) / np.std(self._exp_zarr_data)
            print(f"Loaded experimental zarr with shape {self._exp_zarr_data.shape}")
        except Exception as e:
            print(f"Error loading experimental zarr: {str(e)}")
            raise

    def _load_synthetic_zarr(self):
        """Load synthetic tomogram into memory."""
        try:
            # Get available tomograms from synthetic dataset
            synth_tomograms = self._get_available_tomograms(self.synth_root, self.voxel_spacing)

            if not synth_tomograms:
                raise ValueError(f"No synthetic tomograms found with voxel spacing {self.voxel_spacing}")

            # Select the first tomogram
            synth_tomogram_obj = synth_tomograms[0]
            synth_zarr = zarr.open(synth_tomogram_obj.zarr(), "r")
            self._synth_zarr_data = synth_zarr["0"][:]

            # Normalize tomogram data
            self._synth_zarr_data = (self._synth_zarr_data - np.mean(self._synth_zarr_data)) / np.std(
                self._synth_zarr_data,
            )
            print(f"Loaded synthetic zarr with shape {self._synth_zarr_data.shape}")
        except Exception as e:
            print(f"Error loading synthetic zarr: {str(e)}")
            raise

    def _get_available_tomograms(self, root, voxel_spacing, tomo_type="wbp"):
        """Get available tomograms from a CoPick dataset for a specific voxel spacing."""
        available_tomograms = []

        # If a filtered run is specified, only use that run
        if hasattr(root, "_filtered_run") and root._filtered_run is not None:
            runs = [root._filtered_run]
            print(f"Using only filtered run: {root._filtered_run.meta.name}")
        else:
            runs = root.runs

        for run in runs:
            # Get the closest voxel spacing to the target
            closest_vs = None
            min_diff = float("inf")

            for vs in run.voxel_spacings:
                diff = abs(vs.meta.voxel_size - voxel_spacing)
                if diff < min_diff:
                    min_diff = diff
                    closest_vs = vs

            if closest_vs:
                tomograms = closest_vs.get_tomograms(tomo_type)
                if tomograms:
                    available_tomograms.extend(tomograms)
                    print(
                        f"Found {len(tomograms)} tomograms in run {run.meta.name} with voxel spacing {closest_vs.meta.voxel_size}",
                    )

        return available_tomograms

    def _load_segmentation_masks(self):
        """Load segmentation masks from synthetic dataset into memory."""
        try:
            # Get segmentation masks from synthetic dataset
            segmentation_masks = self._get_segmentation_masks(self.synth_root, self.voxel_spacing)

            if not segmentation_masks:
                raise ValueError(f"No segmentation masks found with voxel spacing {self.voxel_spacing}")

            # Load each mask into memory
            for mask_name, mask_obj in segmentation_masks.items():
                # Skip membrane segmentation masks
                if mask_name.lower() == "membrane":
                    print(f"Skipping membrane segmentation mask: {mask_name}")
                    continue

                # Access the mask data
                mask_zarr = zarr.open(mask_obj.zarr(), "r")
                mask_data = mask_zarr["data" if "data" in mask_zarr else "0"][:]

                # Store the mask data
                self._synth_mask_data[mask_name] = mask_data
                print(f"Loaded mask '{mask_name}' with shape {mask_data.shape}")

            print(f"Loaded {len(self._synth_mask_data)} segmentation masks")
        except Exception as e:
            print(f"Error loading segmentation masks: {str(e)}")
            raise

    def _get_segmentation_masks(self, root, voxel_spacing, pickable_objects=None):
        """Get segmentation masks from a CoPick dataset for a specific voxel spacing."""
        segmentation_masks = {}

        # If a filtered run is specified, only use that run
        if hasattr(root, "_filtered_run") and root._filtered_run is not None:
            runs = [root._filtered_run]
            print(f"Using only filtered run: {root._filtered_run.meta.name} for segmentation masks")
        else:
            runs = root.runs

        for run in runs:
            # Get the closest voxel spacing to the target
            closest_vs = None
            min_diff = float("inf")

            for vs in run.voxel_spacings:
                diff = abs(vs.meta.voxel_size - voxel_spacing)
                if diff < min_diff:
                    min_diff = diff
                    closest_vs = vs

            if closest_vs:
                segmentations = run.get_segmentations(voxel_size=closest_vs.meta.voxel_size)

                for seg in segmentations:
                    # Only include segmentations matching requested pickable objects
                    if pickable_objects is None or seg.meta.name in pickable_objects:
                        segmentation_masks[seg.meta.name] = seg
                        print(f"Found segmentation mask for '{seg.meta.name}' in run {run.meta.name}")

        return segmentation_masks

    def _extract_random_crop(self, tomogram_data, crop_size):
        """Extract a random crop from a tomogram."""
        depth, height, width = tomogram_data.shape

        # Ensure crop sizes don't exceed tomogram dimensions
        crop_depth = min(crop_size[0], depth)
        crop_height = min(crop_size[1], height)
        crop_width = min(crop_size[2], width)

        # Calculate valid ranges for the random crop
        max_z = depth - crop_depth
        max_y = height - crop_height
        max_x = width - crop_width

        if max_z <= 0 or max_y <= 0 or max_x <= 0:
            # Tomogram is smaller than crop size in at least one dimension
            return resize(tomogram_data, crop_size, mode="reflect", anti_aliasing=True)

        # Get random start coordinates
        z_start = random.randint(0, max_z)
        y_start = random.randint(0, max_y)
        x_start = random.randint(0, max_x)

        # Extract the crop
        crop = tomogram_data[
            z_start : z_start + crop_depth,
            y_start : y_start + crop_height,
            x_start : x_start + crop_width,
        ]

        return crop

    def _extract_bounding_box(self, mask_data, mask_name):
        """Extract a bounding box for a connected component in a segmentation mask."""
        # Label connected components
        labels = measure.label(mask_data > 0)
        regions = measure.regionprops(labels)

        if not regions:
            return None

        # Select a random region to extract
        region = random.choice(regions)

        # Get the centroid of the region
        z_center, y_center, x_center = region.centroid

        # Calculate box boundaries centered on the particle
        box_size = self.boxsize[0]  # Assume cubic box
        half_size = box_size // 2

        z_min = max(0, int(z_center - half_size))
        y_min = max(0, int(y_center - half_size))
        x_min = max(0, int(x_center - half_size))

        # Adjust if box would go beyond bounds
        if z_min + box_size > mask_data.shape[0]:
            z_min = max(0, mask_data.shape[0] - box_size)
        if y_min + box_size > mask_data.shape[1]:
            y_min = max(0, mask_data.shape[1] - box_size)
        if x_min + box_size > mask_data.shape[2]:
            x_min = max(0, mask_data.shape[2] - box_size)

        # Calculate max coordinates
        z_max = min(mask_data.shape[0], z_min + box_size)
        y_max = min(mask_data.shape[1], y_min + box_size)
        x_max = min(mask_data.shape[2], x_min + box_size)

        # Check if we can extract a full box
        if (z_max - z_min) != box_size or (y_max - y_min) != box_size or (x_max - x_min) != box_size:
            print(f"Cannot extract a full {box_size}^3 box at the edge of the volume.")
            return None

        # Create a mask for this specific region
        region_mask = np.zeros(mask_data.shape, dtype=bool)
        region_mask[labels == region.label] = True

        # Dilate the mask slightly for smoother boundaries
        dilated_mask = binary_dilation(region_mask, iterations=2)

        # Extract the fixed-size box from the mask
        box_mask = dilated_mask[z_min:z_max, y_min:y_max, x_min:x_max].copy()

        # Verify box mask has expected dimensions
        if box_mask.shape != (box_size, box_size, box_size):
            print(f"Box mask has unexpected shape: {box_mask.shape}")
            return None

        # Extract corresponding region from synthetic tomogram
        try:
            synth_region = self._synth_zarr_data[z_min:z_max, y_min:y_max, x_min:x_max].copy()
            if synth_region.shape != box_mask.shape:
                synth_region = resize(synth_region, box_mask.shape, mode="reflect", anti_aliasing=True)
        except Exception as e:
            print(f"Error extracting synthetic region: {e}")
            return None

        # Return the bounding box info
        return {
            "bbox": (z_min, y_min, x_min, z_max, y_max, x_max),
            "region_mask": box_mask,
            "synth_region": synth_region,
            "object_name": mask_name,
            "center": region.centroid,
        }

    def _splice_volumes(self, synthetic_region, region_mask, exp_crop):
        """Splice a synthetic structure into an experimental tomogram using Gaussian blending at the edges."""
        # Create a spliced volume by starting with the experimental crop
        spliced_volume = exp_crop.copy()

        # For Gaussian blending, create a weight map that transitions smoothly from 1 to 0
        if self.blend_sigma > 0:
            try:
                # Start with the region mask
                mask_float = region_mask.astype(np.float32)

                # Apply Gaussian blur to the mask to create a smooth transition at the boundaries
                # This creates a weight map that goes from 1 (inside) to 0 (outside) with smooth transitions
                weight_map = gaussian_filter(mask_float, sigma=self.blend_sigma)

                # Normalize weight map to ensure it's between 0 and 1
                weight_map = np.clip(weight_map, 0, 1)

                # Apply weighted blending: synthetic * weight + experimental * (1-weight)
                spliced_volume = synthetic_region * weight_map + exp_crop * (1 - weight_map)

            except Exception as e:
                print(f"Error during Gaussian blending at boundaries: {e}")
                # Fall back to simple mask-based splicing
                spliced_volume[region_mask] = synthetic_region[region_mask]
        else:
            # If blend_sigma is 0, just do simple mask-based splicing without blending
            spliced_volume[region_mask] = synthetic_region[region_mask]

        return spliced_volume

    def __getitem__(self, idx):
        """Get an item with spliced mixup augmentation."""
        # Ensure zarr data is loaded
        self._ensure_zarr_loaded()

        # Get the base subvolume using the parent method (without normalizing and tensor conversion)
        subvolume = self._subvolumes[idx].copy()
        molecule_idx = self._molecule_ids[idx]

        # Get a random mask and extract a bounding box
        mask_name = random.choice(list(self._synth_mask_data.keys()))
        bbox_info = self._extract_bounding_box(self._synth_mask_data[mask_name], mask_name)

        if bbox_info is not None:
            # Extract a random crop from the experimental tomogram
            exp_crop = self._extract_random_crop(self._exp_zarr_data, self.boxsize)

            # Splice the volumes
            spliced_volume = self._splice_volumes(bbox_info["synth_region"], bbox_info["region_mask"], exp_crop)

            # Decide whether to use the spliced volume based on a random chance
            if random.random() < 0.5:
                subvolume = spliced_volume
                # Get the molecule_idx for the synthetic object
                if bbox_info["object_name"] not in self._keys:
                    self._keys.append(bbox_info["object_name"])
                molecule_idx = self._keys.index(bbox_info["object_name"])

        # Apply augmentations if enabled
        if self.augment:
            subvolume = self._augment_subvolume(subvolume)

        # Normalize subvolume
        subvolume = (subvolume - np.mean(subvolume)) / (np.std(subvolume) + 1e-6)

        # Add channel dimension and convert to tensor
        subvolume = torch.as_tensor(subvolume[None, ...], dtype=torch.float32)

        # Return the subvolume and class index as a simple tuple
        return subvolume, molecule_idx
