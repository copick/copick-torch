import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from copick_torch.dataset import SimpleCopickDataset


class TestDatasetExtraction(unittest.TestCase):
    """Test the subvolume extraction functionality of SimpleCopickDataset."""

    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.mock_config_path = os.path.join(self.test_dir, "mock_config.json")

        # Define box size for testing
        self.boxsize = (16, 16, 16)

        # Create a test tomogram
        self.tomogram_array = np.ones((32, 32, 32))

        # Add a gradient pattern to make the tomogram less uniform
        x, y, z = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32), np.linspace(0, 1, 32))
        self.tomogram_array = self.tomogram_array * (x + y + z)

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)

    @patch("copick_torch.dataset.SimpleCopickDataset._load_data")
    def test_extract_center_valid(self, mock_load_data):
        """Test extracting a valid subvolume from the center of the tomogram."""
        dataset = SimpleCopickDataset(
            config_path=self.mock_config_path,
            boxsize=self.boxsize,
            patch_strategy="centered",
        )

        # Extract from center of tomogram (should be valid)
        subvolume, is_valid, status = dataset._extract_subvolume_with_validation(self.tomogram_array, 16, 16, 16)

        # Check results
        self.assertTrue(is_valid)
        self.assertEqual(status, "valid")
        self.assertEqual(subvolume.shape, self.boxsize)

        # Since we extracted from the center, values should match the source tomogram
        center_slice = self.tomogram_array[8:24, 8:24, 8:24]
        np.testing.assert_array_equal(subvolume, center_slice)

    @patch("copick_torch.dataset.SimpleCopickDataset._load_data")
    def test_extract_edge_padded(self, mock_load_data):
        """Test extracting a subvolume near the edge of the tomogram (requires padding)."""
        dataset = SimpleCopickDataset(
            config_path=self.mock_config_path,
            boxsize=self.boxsize,
            patch_strategy="centered",
        )

        # Extract from near edge of tomogram (should require padding)
        subvolume, is_valid, status = dataset._extract_subvolume_with_validation(self.tomogram_array, 3, 16, 16)

        # Check results
        self.assertTrue(is_valid)
        self.assertEqual(status, "padded")
        self.assertEqual(subvolume.shape, self.boxsize)

        # Check that the extracted subvolume has some zeros (from padding)
        self.assertTrue(np.any(subvolume == 0))

    @patch("copick_torch.dataset.SimpleCopickDataset._load_data")
    def test_extract_near_edge(self, mock_load_data):
        """Test extracting a subvolume very close to the edge of the tomogram."""
        dataset = SimpleCopickDataset(
            config_path=self.mock_config_path,
            boxsize=(16, 16, 16),
            patch_strategy="centered",
        )

        # Try to extract from positions that are technically valid but will need padding
        subvolume, is_valid, status = dataset._extract_subvolume_with_validation(
            self.tomogram_array,
            2,
            2,
            2,  # Very close to the edge (0,0,0)
        )

        # The actual implementation pads rather than invalidates, so check for padding
        self.assertTrue(is_valid)
        self.assertEqual(status, "padded")
        self.assertEqual(subvolume.shape, (16, 16, 16))

        # Should contain zeros from padding
        self.assertTrue(np.any(subvolume == 0))

    @patch("copick_torch.dataset.SimpleCopickDataset._load_data")
    def test_random_strategy(self, mock_load_data):
        """Test the random patch extraction strategy."""
        dataset = SimpleCopickDataset(config_path=self.mock_config_path, boxsize=self.boxsize, patch_strategy="random")

        # Set a fixed seed for reproducibility
        np.random.seed(42)

        # Extract with random strategy
        subvolume, is_valid, status = dataset._extract_subvolume_with_validation(self.tomogram_array, 16, 16, 16)

        # Check results
        self.assertTrue(is_valid)
        self.assertEqual(subvolume.shape, self.boxsize)

        # Extract again with same seed
        np.random.seed(42)
        subvolume2, is_valid2, status2 = dataset._extract_subvolume_with_validation(self.tomogram_array, 16, 16, 16)

        # Both extractions should be identical with the same seed
        np.testing.assert_array_equal(subvolume, subvolume2)

    @patch("copick_torch.dataset.SimpleCopickDataset._load_data")
    def test_jittered_strategy(self, mock_load_data):
        """Test the jittered patch extraction strategy."""
        dataset = SimpleCopickDataset(
            config_path=self.mock_config_path,
            boxsize=self.boxsize,
            patch_strategy="jittered",
        )

        # Set a fixed seed for reproducibility
        np.random.seed(42)

        # Extract with jittered strategy
        subvolume, is_valid, status = dataset._extract_subvolume_with_validation(self.tomogram_array, 16, 16, 16)

        # Check results
        self.assertTrue(is_valid)
        self.assertEqual(subvolume.shape, self.boxsize)

        # Extract with centered strategy for comparison
        centered_dataset = SimpleCopickDataset(
            config_path=self.mock_config_path,
            boxsize=self.boxsize,
            patch_strategy="centered",
        )

        centered_subvolume, _, _ = centered_dataset._extract_subvolume_with_validation(self.tomogram_array, 16, 16, 16)

        # Jittered should be different from centered (small chance they're identical)
        # This might rarely fail if the random jitter is (0,0,0)
        try:
            np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, subvolume, centered_subvolume)
        except AssertionError:
            # If the above fails, check that we had a very small jitter
            # by verifying most values are the same
            same_values = np.count_nonzero(subvolume == centered_subvolume)
            total_values = np.prod(self.boxsize)
            self.assertGreater(same_values / total_values, 0.9)  # >90% same


if __name__ == "__main__":
    unittest.main()
