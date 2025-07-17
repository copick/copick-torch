"""
A minimal CopickDataset implementation without caching or augmentation.
"""

import json
import logging
import os
from collections import Counter
from types import SimpleNamespace

import copick
import numpy as np
import torch
import zarr
from torch.utils.data import Dataset
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

    This dataset can be saved to disk and loaded later for reproducibility.
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
        min_background_distance=None,
        preload=True,
    ):
        """
        Initialize a MinimalCopickDataset.

        Args:
            proj: A copick project object. If provided, dataset_id and overlay_root are ignored.
            dataset_id: Dataset ID from the CZ cryoET Data Portal. Only used if proj is None.
            overlay_root: Root directory for the overlay storage. Only used if proj is None.
            boxsize: Size of the subvolumes to extract (z, y, x)
            voxel_spacing: Voxel spacing to use for extraction
            include_background: Whether to include background samples
            background_ratio: Ratio of background to particle samples
            min_background_distance: Minimum distance from particles for background samples
            preload: Whether to preload all subvolumes into memory (faster but more memory intensive)
        """
        self.dataset_id = dataset_id
        self.overlay_root = overlay_root
        self.boxsize = boxsize
        self.voxel_spacing = voxel_spacing
        self.include_background = include_background
        self.background_ratio = background_ratio
        self.min_background_distance = min_background_distance or max(boxsize)
        self.preload = preload

        # Initialize data structures
        self._points = []  # List of (x, y, z) coordinates
        self._labels = []  # List of class indices
        self._is_background = []  # List of booleans indicating if a sample is background
        self._tomogram_data = []  # List of tomogram zarr arrays
        self._name_to_label = {}  # Mapping from object names to labels

        # Storage for preloaded data
        self._subvolumes = None

        # Set copick project
        self.copick_root = proj

        # Load the data
        if self.copick_root is not None:
            self._load_data()
        elif dataset_id is not None and overlay_root is not None:
            # Create project from dataset_id and overlay_root
            self.copick_root = copick.from_czcdp_datasets([dataset_id], overlay_root=overlay_root)
            self._load_data()

    def _extract_name_to_label(self):
        """Extract name to label mapping from pickable objects."""
        # Create mapping from object names to labels
        self._name_to_label = {}
        for obj in self.copick_root.pickable_objects:
            self._name_to_label[obj.name] = obj.label

        # Ensure we have a consistent list of object names
        self._object_names = list(self._name_to_label.keys())

        logger.info(f"Name to label mapping: {self._name_to_label}")

    def _load_data(self):
        """Load data from the copick project."""
        try:
            # Extract name to label mapping
            self._extract_name_to_label()

            # Process each run
            all_points = []
            all_labels = []
            all_is_background = []
            all_tomogram_indices = []

            for run_idx, run in enumerate(self.copick_root.runs):
                logger.info(f"Processing run: {run.name}")

                # Get tomogram
                try:
                    voxel_spacing_obj = run.get_voxel_spacing(self.voxel_spacing)
                    if not voxel_spacing_obj or not voxel_spacing_obj.tomograms:
                        logger.warning(f"No tomograms found for run {run.name} at voxel spacing {self.voxel_spacing}")
                        continue

                    # Find a denoised tomogram if available, otherwise use the first one
                    tomogram = [t for t in voxel_spacing_obj.tomograms if "wbp-denoised" in t.tomo_type]
                    if not tomogram:
                        tomogram = voxel_spacing_obj.tomograms[0]
                    else:
                        tomogram = tomogram[0]

                    # Open zarr array and load it fully into memory
                    tomogram_zarr = zarr.open(tomogram.zarr())["0"]
                    tomogram_data = np.array(tomogram_zarr[:])
                    self._tomogram_data.append(tomogram_data)
                    logger.info(f"Loaded tomogram with shape {tomogram_data.shape} into memory")

                    # Store all particle coordinates for background sampling
                    all_particle_coords = []

                    # Initialize storage for preloaded data if preloading is enabled
                    if self.preload and not hasattr(self, "_subvolumes"):
                        self._subvolumes = []

                    # Process picks for each object type
                    for picks in run.get_picks():
                        if not picks.from_tool:
                            continue

                        object_name = picks.pickable_object_name

                        # Skip objects not in our mapping
                        if object_name not in self._name_to_label:
                            logger.warning(f"Object {object_name} not in pickable objects, skipping")
                            continue

                        class_idx = self._name_to_label[object_name]

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
                                all_tomogram_indices.append(run_idx)
                                all_particle_coords.append(point)

                                # If preloading is enabled, extract and store the subvolume immediately
                                if self.preload:
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
                                    x_end = min(tomogram_data.shape[2], x_idx + half_x)
                                    y_start = max(0, y_idx - half_y)
                                    y_end = min(tomogram_data.shape[1], y_idx + half_y)
                                    z_start = max(0, z_idx - half_z)
                                    z_end = min(tomogram_data.shape[0], z_idx + half_z)

                                    # Extract subvolume
                                    subvolume = tomogram_data[z_start:z_end, y_start:y_end, x_start:x_end].copy()

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
                                            z_offset : z_offset + pad_z,
                                            y_offset : y_offset + pad_y,
                                            x_offset : x_offset + pad_x,
                                        ] = subvolume

                                        subvolume = padded

                                    # Normalize
                                    if np.std(subvolume) > 0:
                                        subvolume = (subvolume - np.mean(subvolume)) / np.std(subvolume)

                                    # Add channel dimension and convert to tensor
                                    subvolume_tensor = torch.as_tensor(subvolume[None, ...], dtype=torch.float32)

                                    # Store the preloaded tensor with its label
                                    self._subvolumes.append((subvolume_tensor, class_idx))
                        except Exception as e:
                            logger.error(f"Error processing picks for {object_name}: {e}")

                    # Sample background points if requested
                    if self.include_background and all_particle_coords:
                        num_particles = len(all_particle_coords)
                        num_background = int(num_particles * self.background_ratio)

                        logger.info(f"Sampling {num_background} background points")

                        bg_points = self._sample_background_points(
                            tomogram_data.shape,
                            all_particle_coords,
                            num_background,
                            self.min_background_distance,
                        )

                        for point in bg_points:
                            all_points.append(point)
                            all_labels.append(-1)  # -1 indicates background
                            all_is_background.append(True)
                            all_tomogram_indices.append(run_idx)

                            # If preloading is enabled, extract and store the background subvolume immediately
                            if self.preload:
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
                                x_end = min(tomogram_data.shape[2], x_idx + half_x)
                                y_start = max(0, y_idx - half_y)
                                y_end = min(tomogram_data.shape[1], y_idx + half_y)
                                z_start = max(0, z_idx - half_z)
                                z_end = min(tomogram_data.shape[0], z_idx + half_z)

                                # Extract subvolume
                                subvolume = tomogram_data[z_start:z_end, y_start:y_end, x_start:x_end].copy()

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
                                        z_offset : z_offset + pad_z,
                                        y_offset : y_offset + pad_y,
                                        x_offset : x_offset + pad_x,
                                    ] = subvolume

                                    subvolume = padded

                                # Normalize
                                if np.std(subvolume) > 0:
                                    subvolume = (subvolume - np.mean(subvolume)) / np.std(subvolume)

                                # Add channel dimension and convert to tensor
                                subvolume_tensor = torch.as_tensor(subvolume[None, ...], dtype=torch.float32)

                                # Store the preloaded tensor with its label (-1 for background)
                                self._subvolumes.append((subvolume_tensor, -1))

                except Exception as e:
                    logger.error(f"Error processing tomogram for run {run.name}: {e}")
                    continue

            # Store the processed data
            self._points = all_points
            self._labels = all_labels
            self._is_background = all_is_background
            self._tomogram_indices = all_tomogram_indices

            logger.info(f"Dataset loaded with {len(self._points)} samples")

            # Print class distribution
            self._print_class_distribution()

            # If preloading is enabled, the subvolumes are already preloaded during point extraction
            if self.preload and len(self._points) > 0 and not hasattr(self, "_subvolumes"):
                logger.info("Preloading was requested but no preloaded data exists")

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _preload_data(self):
        """Preload all subvolumes into memory."""
        logger.info(f"Preloading {len(self._points)} subvolumes into memory...")

        # This method is preserved for backward compatibility but should not be called
        # during normal operation since preloading now happens during _load_data

        # Check if preloading already happened
        if hasattr(self, "_subvolumes") and self._subvolumes:
            logger.info(f"Subvolumes are already preloaded ({len(self._subvolumes)} subvolumes)")
            return

        # Initialize storage for preloaded data
        self._subvolumes = []

        # Extract and store all subvolumes
        for idx in tqdm(range(len(self._points))):
            point = self._points[idx]
            label = self._labels[idx]
            tomogram_idx = self._tomogram_indices[idx] if hasattr(self, "_tomogram_indices") else 0

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
            # Find the class name for this label
            for name, label in self._name_to_label.items():
                if label == cls_idx:
                    distribution[name] = count
                    break

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

    def extract_subvolume(self, point, tomogram_idx=0):
        """
        Extract a cubic subvolume centered around a point.

        Args:
            point: (x, y, z) coordinates
            tomogram_idx: Index of the tomogram to use

        Returns:
            Extracted subvolume as a numpy array
        """
        # Check if tomogram exists
        if tomogram_idx >= len(self._tomogram_data) or self._tomogram_data[tomogram_idx] is None:
            raise ValueError(f"No tomogram found at index {tomogram_idx}")

        tomogram_zarr = self._tomogram_data[tomogram_idx]

        # Get dimensions of the tomogram
        z_dim, y_dim, x_dim = tomogram_zarr.shape

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
        subvolume = tomogram_zarr[z_start:z_end, y_start:y_end, x_start:x_end].copy()

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
            padded[z_offset : z_offset + pad_z, y_offset : y_offset + pad_y, x_offset : x_offset + pad_x] = subvolume

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
        # If data is preloaded, return from preloaded data
        if self.preload and hasattr(self, "_subvolumes") and self._subvolumes:
            return self._subvolumes[idx]

        # Otherwise, extract on-the-fly
        # Get the point, label, and tomogram index
        point = self._points[idx]
        label = self._labels[idx]
        tomogram_idx = self._tomogram_indices[idx] if hasattr(self, "_tomogram_indices") else 0

        # Extract the subvolume
        subvolume = self.extract_subvolume(point, tomogram_idx)

        # Normalize
        if np.std(subvolume) > 0:
            subvolume = (subvolume - np.mean(subvolume)) / np.std(subvolume)

        # Add channel dimension and convert to tensor
        subvolume_tensor = torch.as_tensor(subvolume[None, ...], dtype=torch.float32)

        return subvolume_tensor, label

    def keys(self):
        """Get the list of class names."""
        # Add background class if included
        class_names = list(self._name_to_label.keys())
        if self.include_background:
            return class_names + ["background"]
        return class_names

    def get_class_distribution(self):
        """Get the distribution of classes in the dataset."""
        distribution = Counter()

        for label in self._labels:
            if label == -1:
                distribution["background"] += 1
            else:
                # Find the class name for this label
                for name, idx in self._name_to_label.items():
                    if idx == label:
                        distribution[name] += 1
                        break

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

    def save(self, save_dir):
        """
        Save the dataset to disk for later reloading.

        Args:
            save_dir: Directory to save the dataset
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save metadata
        metadata = {
            "dataset_id": self.dataset_id,
            "boxsize": self.boxsize,
            "voxel_spacing": self.voxel_spacing,
            "include_background": self.include_background,
            "background_ratio": self.background_ratio,
            "min_background_distance": self.min_background_distance,
            "name_to_label": self._name_to_label,
            "preload": self.preload,
        }

        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        # If preloaded, save the actual tensors
        if self.preload and hasattr(self, "_subvolumes") and self._subvolumes:
            logger.info("Saving preloaded tensors...")

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
            torch.save(subvolumes_tensor, os.path.join(save_dir, "subvolumes.pt"))
            torch.save(labels_tensor, os.path.join(save_dir, "labels.pt"))

            logger.info(f"Saved {len(subvolumes)} preloaded tensors")
        else:
            # Save sample information for on-the-fly loading
            logger.info("Saving sample information for on-the-fly loading...")

            sample_data = []
            for i in range(len(self._points)):
                point = self._points[i]
                label = self._labels[i]
                is_background = self._is_background[i]
                tomogram_idx = self._tomogram_indices[i] if hasattr(self, "_tomogram_indices") else 0

                sample_data.append(
                    {
                        "point": point.tolist() if isinstance(point, np.ndarray) else point,
                        "label": int(label),
                        "is_background": bool(is_background),
                        "tomogram_idx": int(tomogram_idx),
                    },
                )

            with open(os.path.join(save_dir, "samples.json"), "w") as f:
                json.dump(sample_data, f)

        # Save tomogram information (needed for on-the-fly loading)
        tomogram_info = []
        for idx, tomogram in enumerate(self._tomogram_data):
            tomo_data = {"index": idx, "shape": list(tomogram.shape), "path": getattr(tomogram, "path", str(tomogram))}
            tomogram_info.append(tomo_data)

        with open(os.path.join(save_dir, "tomogram_info.json"), "w") as f:
            json.dump(tomogram_info, f)

        logger.info(f"Dataset saved to {save_dir}")

    @classmethod
    def load(cls, save_dir, proj=None):
        """
        Load a previously saved dataset.

        Args:
            save_dir: Directory where the dataset was saved
            proj: Optional copick project object. If provided, tomograms will be loaded from it.

        Returns:
            Loaded MinimalCopickDataset instance
        """
        # Load metadata
        with open(os.path.join(save_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)

        # Create a new dataset instance without loading data
        dataset = cls.__new__(cls)
        dataset.dataset_id = metadata.get("dataset_id")
        dataset.boxsize = metadata.get("boxsize", (48, 48, 48))
        dataset.voxel_spacing = metadata.get("voxel_spacing", 10.012)
        dataset.include_background = metadata.get("include_background", False)
        dataset.background_ratio = metadata.get("background_ratio", 0.2)
        dataset.min_background_distance = metadata.get("min_background_distance")
        dataset._name_to_label = metadata.get("name_to_label", {})
        dataset.preload = metadata.get("preload", True)
        dataset.copick_root = proj

        # Check if we have preloaded tensors
        subvolumes_path = os.path.join(save_dir, "subvolumes.pt")
        labels_path = os.path.join(save_dir, "labels.pt")

        if os.path.exists(subvolumes_path) and os.path.exists(labels_path):
            logger.info("Loading preloaded tensors...")

            # Load the tensors
            subvolumes = torch.load(subvolumes_path)
            labels = torch.load(labels_path)

            # Store in the dataset
            dataset._subvolumes = [(subvolumes[i], labels[i].item()) for i in range(len(labels))]

            # Create minimal point/label data for compatibility
            dataset._points = [np.zeros(3) for _ in range(len(labels))]
            dataset._labels = [label.item() for label in labels]
            dataset._is_background = [label.item() == -1 for label in labels]
            dataset._tomogram_indices = [0 for _ in range(len(labels))]
            dataset._tomogram_data = []

            logger.info(f"Loaded dataset with {len(dataset._subvolumes)} preloaded subvolumes")
        else:
            # Initialize empty data structures
            dataset._tomogram_data = []

            # Load sample information
            with open(os.path.join(save_dir, "samples.json"), "r") as f:
                sample_data = json.load(f)

            # Extract sample information
            dataset._points = [np.array(s["point"]) for s in sample_data]
            dataset._labels = [s["label"] for s in sample_data]
            dataset._is_background = [s["is_background"] for s in sample_data]
            dataset._tomogram_indices = [s["tomogram_idx"] for s in sample_data]

            # Load tomogram information
            with open(os.path.join(save_dir, "tomogram_info.json"), "r") as f:
                tomogram_info = json.load(f)

            # If a project is provided, attempt to load tomograms from it
            if proj is not None:
                logger.info("Loading tomograms from provided project")

                # Initialize tomogram list with placeholders
                dataset._tomogram_data = [None] * len(tomogram_info)

                # Attempt to load tomograms
                for run in proj.runs:
                    voxel_spacing_obj = run.get_voxel_spacing(dataset.voxel_spacing)
                    if voxel_spacing_obj and voxel_spacing_obj.tomograms:
                        # Find denoised tomogram if available
                        tomogram = [t for t in voxel_spacing_obj.tomograms if "wbp-denoised" in t.tomo_type]
                        if not tomogram:
                            tomogram = voxel_spacing_obj.tomograms[0]
                        else:
                            tomogram = tomogram[0]

                        # Find matching tomogram in info
                        for tomo_info in tomogram_info:
                            # Simple check for matching shape as a heuristic
                            tomo_zarr = zarr.open(tomogram.zarr())["0"]
                            if list(tomo_zarr.shape) == tomo_info["shape"]:
                                idx = tomo_info["index"]
                                dataset._tomogram_data[idx] = tomo_zarr
                                logger.info(f"Loaded tomogram at index {idx} with shape {tomo_zarr.shape}")
            else:
                logger.warning("No project provided. Tomograms must be loaded separately.")

            logger.info(f"Loaded dataset with {len(dataset._points)} samples")

            # If preload is True and we have tomogram data, preload the subvolumes
            if dataset.preload and dataset._tomogram_data and all(t is not None for t in dataset._tomogram_data):
                logger.info("Preloading subvolumes...")
                dataset._preload_data()

        # Print class distribution
        dataset._print_class_distribution()

        return dataset
