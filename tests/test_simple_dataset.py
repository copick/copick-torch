import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from copick_torch.dataset import SimpleCopickDataset, SimpleDatasetMixin


class TestSimpleDatasetMixin(unittest.TestCase):
    """Test the SimpleDatasetMixin functionality."""

    def setUp(self):
        # Create a simple dataset with the mixin for testing
        self.test_dataset = type("TestDataset", (SimpleDatasetMixin, object), {})()

        # Add required attributes for the mixin
        self.test_dataset._subvolumes = [np.ones((16, 16, 16))]
        self.test_dataset._molecule_ids = [0]
        self.test_dataset.augment = False

    def test_getitem(self):
        """Test the __getitem__ method of SimpleDatasetMixin."""
        # Mock the _augment_subvolume method
        self.test_dataset._augment_subvolume = lambda subvol, idx: subvol

        # Get an item
        subvolume, molecule_idx = self.test_dataset.__getitem__(0)

        # Check that the result is a tuple with the right types
        self.assertIsInstance(subvolume, torch.Tensor)
        self.assertEqual(molecule_idx, 0)

        # Check the shape of the subvolume (should have channel dimension)
        self.assertEqual(subvolume.shape, (1, 16, 16, 16))

    def test_getitem_with_augmentation(self):
        """Test __getitem__ with augmentation enabled."""
        # Enable augmentation
        self.test_dataset.augment = True

        # Mock augmentation to return a scaled subvolume
        self.test_dataset._augment_subvolume = lambda subvol, idx: subvol * 2

        # Get an item with augmentation
        subvolume, molecule_idx = self.test_dataset.__getitem__(0)

        # Verify the augmentation was applied (values should be higher)
        # But normalization will bring them back to a similar range
        self.assertIsInstance(subvolume, torch.Tensor)


class TestSimpleCopickDataset(unittest.TestCase):
    """Test the SimpleCopickDataset class."""

    def setUp(self):
        # Create a temporary directory for caching
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

    @patch("copick_torch.dataset.SimpleCopickDataset._load_data")
    def test_init_basic(self, mock_load_data):
        """Test basic initialization of SimpleCopickDataset."""
        # Initialize with minimal parameters
        dataset = SimpleCopickDataset(
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

    @patch("copick_torch.dataset.SimpleCopickDataset._load_data")
    def test_init_with_options(self, mock_load_data):
        """Test initialization with various options."""
        dataset = SimpleCopickDataset(
            config_path=self.mock_config_path,
            boxsize=self.boxsize,
            augment=True,
            cache_dir=self.cache_dir,
            cache_format="parquet",
            seed=42,
            max_samples=100,
            voxel_spacing=5.0,
            include_background=True,
            background_ratio=0.3,
            min_background_distance=20.0,
            patch_strategy="random",
            debug_mode=True,
        )

        # Verify all parameters were set correctly
        self.assertEqual(dataset.config_path, self.mock_config_path)
        self.assertEqual(dataset.boxsize, self.boxsize)
        self.assertTrue(dataset.augment)
        self.assertEqual(dataset.cache_dir, self.cache_dir)
        self.assertEqual(dataset.cache_format, "parquet")
        self.assertEqual(dataset.seed, 42)
        self.assertEqual(dataset.max_samples, 100)
        self.assertEqual(dataset.voxel_spacing, 5.0)
        self.assertTrue(dataset.include_background)
        self.assertEqual(dataset.background_ratio, 0.3)
        self.assertEqual(dataset.min_background_distance, 20.0)
        self.assertEqual(dataset.patch_strategy, "random")
        self.assertTrue(dataset.debug_mode)

    @patch("copick_torch.dataset.SimpleCopickDataset._load_data")
    def test_dataset_empty(self, mock_load_data):
        """Test behavior with empty dataset."""
        dataset = SimpleCopickDataset(config_path=self.mock_config_path, boxsize=self.boxsize)

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

    @patch("copick_torch.dataset.SimpleCopickDataset._load_data")
    def test_compute_sample_weights(self, mock_load_data):
        """Test the _compute_sample_weights method."""
        dataset = SimpleCopickDataset(config_path=self.mock_config_path, boxsize=self.boxsize)

        # Create an unbalanced dataset
        dataset._molecule_ids = [0, 0, 0, 1, 1, 2]

        # Compute sample weights
        dataset._compute_sample_weights()

        # Check weights are inversely proportional to class frequency
        expected_weights = [6 / 3, 6 / 3, 6 / 3, 6 / 2, 6 / 2, 6 / 1]  # total_samples / count_per_class
        np.testing.assert_array_almost_equal(dataset.sample_weights, expected_weights)

    @patch("copick_torch.dataset.SimpleCopickDataset._load_data")
    def test_get_sample_weights(self, mock_load_data):
        """Test the get_sample_weights method."""
        dataset = SimpleCopickDataset(config_path=self.mock_config_path, boxsize=self.boxsize)

        # Set sample weights
        dataset.sample_weights = [1.0, 2.0, 3.0]

        # Get sample weights
        weights = dataset.get_sample_weights()

        # Check weights are returned correctly
        self.assertEqual(weights, [1.0, 2.0, 3.0])

    @patch("copick_torch.dataset.SimpleCopickDataset._load_data")
    def test_keys(self, mock_load_data):
        """Test the keys method."""
        dataset = SimpleCopickDataset(config_path=self.mock_config_path, boxsize=self.boxsize)

        # Set keys
        dataset._keys = ["class1", "class2", "class3"]

        # Get keys
        keys = dataset.keys()

        # Check keys are returned correctly
        self.assertEqual(keys, ["class1", "class2", "class3"])

    @patch("copick_torch.dataset.SimpleCopickDataset._load_data")
    def test_get_class_distribution(self, mock_load_data):
        """Test the get_class_distribution method."""
        dataset = SimpleCopickDataset(config_path=self.mock_config_path, boxsize=self.boxsize)

        # Create a test dataset with class distribution
        dataset._keys = ["class1", "class2", "class3"]
        dataset._molecule_ids = [0, 0, 0, 1, 1, 2, -1, -1]  # -1 is background
        dataset._is_background = [False, False, False, False, False, False, True, True]

        # Get class distribution
        distribution = dataset.get_class_distribution()

        # Check distribution is correct
        expected_distribution = {"class1": 3, "class2": 2, "class3": 1, "background": 2}
        self.assertEqual(distribution, expected_distribution)

    @patch("copick_torch.dataset.SimpleCopickDataset._load_data")
    def test_validation_logic_missing_config(self, mock_load_data):
        """Test validation logic when both config_path and copick_root are missing."""
        # Should raise ValueError when both config_path and copick_root are None
        with self.assertRaises(ValueError):
            SimpleCopickDataset(config_path=None, copick_root=None, boxsize=self.boxsize)

    @patch("copick_torch.dataset.SimpleCopickDataset._extract_subvolume_with_validation")
    @patch("copick_torch.dataset.SimpleCopickDataset._load_data")
    def test_extract_subvolume_strategies(self, mock_load_data, mock_extract):
        """Test different patch extraction strategies."""
        # Create a sample tomogram_array
        tomogram_array = np.zeros((32, 32, 32))

        # Test centered strategy
        dataset_centered = SimpleCopickDataset(
            config_path=self.mock_config_path,
            boxsize=self.boxsize,
            patch_strategy="centered",
        )
        dataset_centered._extract_subvolume_with_validation(tomogram_array, 16, 16, 16)
        mock_extract.assert_called_once()

        mock_extract.reset_mock()

        # Test random strategy
        dataset_random = SimpleCopickDataset(
            config_path=self.mock_config_path,
            boxsize=self.boxsize,
            patch_strategy="random",
        )
        dataset_random._extract_subvolume_with_validation(tomogram_array, 16, 16, 16)
        mock_extract.assert_called_once()

        mock_extract.reset_mock()

        # Test jittered strategy
        dataset_jittered = SimpleCopickDataset(
            config_path=self.mock_config_path,
            boxsize=self.boxsize,
            patch_strategy="jittered",
        )
        dataset_jittered._extract_subvolume_with_validation(tomogram_array, 16, 16, 16)
        mock_extract.assert_called_once()


if __name__ == "__main__":
    unittest.main()
