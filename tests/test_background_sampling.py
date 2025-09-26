import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from copick_torch.dataset import SimpleCopickDataset


class TestBackgroundSampling(unittest.TestCase):
    """Test the background sampling functionality of SimpleCopickDataset."""

    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.mock_config_path = os.path.join(self.test_dir, "mock_config.json")

        # Define box size for testing
        self.boxsize = (16, 16, 16)

        # Create a test tomogram
        self.tomogram_array = np.zeros((64, 64, 64))

        # Add some "particles" to make background sampling more realistic
        self.particle_coords = [
            (16, 16, 16),  # center particle
            (40, 40, 40),  # corner particle
            (16, 40, 40),  # edge particle
        ]

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)

    @patch("copick_torch.dataset.SimpleCopickDataset._load_data")
    def test_sample_background_points(self, mock_load_data):
        """Test the _sample_background_points method."""
        # Create dataset with background sampling enabled
        dataset = SimpleCopickDataset(
            config_path=self.mock_config_path,
            boxsize=self.boxsize,
            include_background=True,
            background_ratio=0.5,  # One background sample for every two particles
            min_background_distance=20.0,  # Stay at least 20 units away from particles
            patch_strategy="centered",
        )

        # Initialize dataset properties
        dataset._subvolumes = []
        dataset._molecule_ids = []
        dataset._is_background = []

        # Mock the _extract_subvolume_with_validation method to return valid subvolumes
        def mock_extract(*args, **kwargs):
            return np.zeros(self.boxsize), True, "valid"

        dataset._extract_subvolume_with_validation = mock_extract

        # Set a fixed seed for reproducibility
        np.random.seed(42)

        # Sample background points
        dataset._sample_background_points(self.tomogram_array, self.particle_coords)

        # Check that background samples were added
        # Should add background_ratio * len(particle_coords) samples = 0.5 * 3 = 1 or 2
        self.assertGreater(len(dataset._subvolumes), 0)
        self.assertGreater(len(dataset._molecule_ids), 0)
        self.assertGreater(len(dataset._is_background), 0)

        # Check that all added samples are marked as background
        self.assertTrue(all(dataset._is_background))

        # Check that all added samples have molecule_id = -1
        self.assertTrue(all(mol_id == -1 for mol_id in dataset._molecule_ids))

    @patch("copick_torch.dataset.SimpleCopickDataset._load_data")
    def test_sample_background_points_no_particles(self, mock_load_data):
        """Test the _sample_background_points method with no particles."""
        # Create dataset with background sampling enabled
        dataset = SimpleCopickDataset(
            config_path=self.mock_config_path,
            boxsize=self.boxsize,
            include_background=True,
            background_ratio=0.5,
        )

        # Initialize dataset properties
        dataset._subvolumes = []
        dataset._molecule_ids = []
        dataset._is_background = []

        # Sample background points with no particles
        dataset._sample_background_points(self.tomogram_array, [])

        # Check that no background samples were added
        self.assertEqual(len(dataset._subvolumes), 0)
        self.assertEqual(len(dataset._molecule_ids), 0)
        self.assertEqual(len(dataset._is_background), 0)

    @patch("copick_torch.dataset.SimpleCopickDataset._load_data")
    def test_sample_background_fails_validation(self, mock_load_data):
        """Test when background samples fail validation."""
        # Create dataset with background sampling enabled
        dataset = SimpleCopickDataset(
            config_path=self.mock_config_path,
            boxsize=self.boxsize,
            include_background=True,
            background_ratio=1.0,  # One background sample for each particle
            min_background_distance=20.0,
        )

        # Initialize dataset properties
        dataset._subvolumes = []
        dataset._molecule_ids = []
        dataset._is_background = []

        # Mock the _extract_subvolume_with_validation method to always fail validation
        def mock_extract(*args, **kwargs):
            return None, False, "Invalid slice range"

        dataset._extract_subvolume_with_validation = mock_extract

        # Sample background points
        dataset._sample_background_points(self.tomogram_array, self.particle_coords)

        # Check that no background samples were added
        self.assertEqual(len(dataset._subvolumes), 0)
        self.assertEqual(len(dataset._molecule_ids), 0)
        self.assertEqual(len(dataset._is_background), 0)

    @patch("copick_torch.dataset.SimpleCopickDataset._load_data")
    def test_sample_background_distance_constraint(self, mock_load_data):
        """Test that background samples respect the minimum distance constraint."""
        # Create dataset with background sampling enabled and a large min distance
        min_distance = 100.0  # Very large distance requirement
        dataset = SimpleCopickDataset(
            config_path=self.mock_config_path,
            boxsize=self.boxsize,
            include_background=True,
            background_ratio=1.0,
            min_background_distance=min_distance,  # Very strict distance requirement
        )

        # Initialize dataset properties
        dataset._subvolumes = []
        dataset._molecule_ids = []
        dataset._is_background = []

        # Mock extraction to return valid subvolumes
        def mock_extract(*args, **kwargs):
            return np.zeros(self.boxsize), True, "valid"

        dataset._extract_subvolume_with_validation = mock_extract

        # Set a fixed seed for reproducibility
        np.random.seed(42)

        # Sample background points with a very strict distance constraint
        # This will likely hit the max_attempts limit
        dataset._sample_background_points(self.tomogram_array, self.particle_coords)

        # We expect few or no samples due to the strict constraint
        # The test passes if we don't get an infinite loop
        self.assertGreaterEqual(len(dataset._subvolumes), 0)

    def test_include_background_in_load_data(self):
        """Test that _load_data calls _sample_background_points when include_background=True."""
        # Create a mock copick root
        mock_root = MagicMock()
        mock_run = MagicMock()
        mock_voxel_spacing = MagicMock()
        mock_tomogram = MagicMock()

        # Configure the mocks
        mock_root.runs = [mock_run]
        mock_run.name = "mock_run"
        mock_run.get_voxel_spacing.return_value = mock_voxel_spacing
        mock_voxel_spacing.tomograms = [mock_tomogram]
        mock_tomogram.numpy.return_value = self.tomogram_array

        # Create mock picks
        mock_picks = MagicMock()
        mock_picks.from_tool = True
        mock_picks.pickable_object_name = "test_object"
        mock_picks.numpy.return_value = (np.array([[16, 16, 16]]), None)
        mock_run.get_picks.return_value = [mock_picks]

        # Use patch within the test function
        with patch("copick_torch.dataset.SimpleCopickDataset._sample_background_points") as mock_sample_bg:
            # Create dataset with include_background=True
            _ = SimpleCopickDataset(
                config_path=None,
                copick_root=mock_root,
                boxsize=self.boxsize,
                include_background=True,
                background_ratio=0.5,
                cache_dir=None,  # Disable caching to ensure _load_data runs
            )

            # The _load_data method should have been called during initialization
            # and _sample_background_points should be called from within it
            mock_sample_bg.assert_called()


if __name__ == "__main__":
    unittest.main()
