import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from copick_torch.copick import CopickDataset


class TestCopickDataset(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.test_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Mock config path
        self.mock_config_path = os.path.join(self.test_dir, "mock_config.json")

        # Parameters for testing
        self.boxsize = (16, 16, 16)
        self.voxel_spacing = 10.0

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)

    @patch("copick_torch.copick.CopickDataset._load_data")
    def test_init_basic(self, mock_load_data):
        """Test basic initialization of CopickDataset."""
        # Initialize with minimal parameters
        dataset = CopickDataset(
            config_path=self.mock_config_path,
            boxsize=self.boxsize,
            cache_dir=None,  # Don't use caching
        )

        # Verify initialization
        self.assertEqual(dataset.config_path, self.mock_config_path)
        self.assertEqual(dataset.boxsize, self.boxsize)
        self.assertFalse(dataset.augment)
        self.assertIsNone(dataset.cache_dir)

        # Verify _load_data was called
        mock_load_data.assert_called_once()

    @patch("copick_torch.copick.CopickDataset._load_data")
    def test_dataset_empty(self, mock_load_data):
        """Test behavior with empty dataset."""
        dataset = CopickDataset(config_path=self.mock_config_path, boxsize=self.boxsize)

        # Mock empty dataset
        dataset._subvolumes = np.array([])
        dataset._molecule_ids = np.array([])
        dataset._is_background = np.array([])
        dataset._keys = []

        # Test length
        self.assertEqual(len(dataset), 0)

        # Test get_class_distribution with empty dataset
        distribution = dataset.get_class_distribution()
        self.assertEqual(distribution, {})

    def test_augmentations(self):
        """Test data augmentation functions."""
        # Create a dataset with mocked _load_data
        with patch("copick_torch.copick.CopickDataset._load_data"):
            dataset = CopickDataset(config_path=self.mock_config_path, boxsize=self.boxsize, augment=True)

        # Create a test volume
        test_volume = np.ones(self.boxsize)

        # Test brightness augmentation
        augmented = dataset._brightness(test_volume)
        self.assertEqual(augmented.shape, test_volume.shape)
        self.assertNotEqual(np.sum(augmented), np.sum(test_volume))

        # Test intensity scaling
        augmented = dataset._intensity_scaling(test_volume)
        self.assertEqual(augmented.shape, test_volume.shape)

        # Test flip
        augmented = dataset._flip(test_volume)
        self.assertEqual(augmented.shape, test_volume.shape)

        # Test rotate
        augmented = dataset._rotate(test_volume)
        self.assertEqual(augmented.shape, test_volume.shape)

    @patch("copick_torch.copick.CopickDataset._load_data")
    def test_getitem_no_augment(self, mock_load_data):
        """Test __getitem__ without augmentation."""
        dataset = CopickDataset(config_path=self.mock_config_path, boxsize=self.boxsize, augment=False)

        # Create a mock dataset with one item
        test_volume = np.ones(self.boxsize)
        dataset._subvolumes = np.array([test_volume])
        dataset._molecule_ids = np.array([0])
        dataset._is_background = np.array([False])
        dataset._keys = ["test_class"]

        # Get the item
        volume, label = dataset[0]

        # Check shapes and types
        self.assertEqual(volume.shape, (1, *self.boxsize))  # Check channel dimension added
        self.assertIsInstance(volume, torch.Tensor)
        self.assertIsInstance(label, dict)  # Verify label is a dictionary
        self.assertEqual(label["class_idx"], 0)  # Check if class_idx is correct

    @patch("copick_torch.copick.CopickDataset._load_data")
    def test_stratified_split(self, mock_load_data):
        """Test stratified_split method."""
        dataset = CopickDataset(config_path=self.mock_config_path, boxsize=self.boxsize)

        # Create a mock dataset with balanced classes
        n_classes = 3
        n_samples_per_class = 10
        test_volumes = []
        test_labels = []

        for class_idx in range(n_classes):
            for _ in range(n_samples_per_class):
                test_volumes.append(np.ones(self.boxsize))
                test_labels.append(class_idx)

        dataset._subvolumes = np.array(test_volumes)
        dataset._molecule_ids = np.array(test_labels)
        dataset._is_background = np.array([False] * len(test_labels))
        dataset._keys = [f"class_{i}" for i in range(n_classes)]

        # Split the dataset
        train_ds, val_ds, test_ds = dataset.stratified_split(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

        # Check split sizes
        self.assertEqual(len(train_ds), int(0.6 * len(dataset)))
        self.assertEqual(len(val_ds), int(0.2 * len(dataset)))
        self.assertEqual(len(test_ds), len(dataset) - len(train_ds) - len(val_ds))

        # Check that each split contains samples from all classes
        train_labels = [dataset._molecule_ids[i] for i in train_ds.indices]
        val_labels = [dataset._molecule_ids[i] for i in val_ds.indices]
        test_labels = [dataset._molecule_ids[i] for i in test_ds.indices]

        for class_idx in range(n_classes):
            self.assertIn(class_idx, train_labels)
            self.assertIn(class_idx, val_labels)
            self.assertIn(class_idx, test_labels)


if __name__ == "__main__":
    unittest.main()
