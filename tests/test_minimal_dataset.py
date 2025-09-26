import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from copick_torch.minimal_dataset import MinimalCopickDataset


class TestMinimalCopickDataset(unittest.TestCase):
    """
    Test the MinimalCopickDataset class.
    """

    pass

    # TODO: Uncomment and fix the tests below

    # @patch("zarr.open")
    # @patch("copick.from_czcdp_datasets")
    # def test_dataset_initialization(self, mock_from_czcdp, mock_zarr_open):
    #     """Test that the dataset can be initialized and returns correct data."""
    #     # Set up mocks
    #     mock_copick_root = MagicMock()
    #     mock_run = MagicMock()
    #     mock_vs = MagicMock()
    #     mock_tomogram = MagicMock()
    #
    #     # Configure mocks
    #     mock_from_czcdp.return_value = mock_copick_root
    #     mock_copick_root.runs = [mock_run]
    #     mock_copick_root.pickable_objects = [MagicMock(name="object1"), MagicMock(name="object2")]
    #
    #     for po in mock_copick_root.pickable_objects:
    #         po.name = po.name
    #
    #     mock_run.name = "test_run"
    #     mock_run.get_voxel_spacing.return_value = mock_vs
    #     mock_vs.tomograms = [mock_tomogram]
    #     mock_tomogram.tomo_type = "wbp-denoised"
    #
    #     # Setup picks for each object
    #     mock_picks_list = []
    #     for po in mock_copick_root.pickable_objects:
    #         mock_pick = MagicMock()
    #         mock_pick.from_tool = True
    #         mock_pick.pickable_object_name = po.name
    #         # Create 5 points for each object
    #         mock_pick.numpy.return_value = (
    #             np.array([[100, 100, 100], [200, 200, 200], [300, 300, 300], [400, 400, 400], [500, 500, 500]]),
    #             None,
    #         )
    #         mock_picks_list.append(mock_pick)
    #
    #     mock_run.get_picks.return_value = mock_picks_list
    #
    #     # Mock zarr.open to return a dummy array
    #     mock_zarr_root = MagicMock()
    #     mock_zarr_open.return_value = mock_zarr_root
    #     # Create a dummy 3D array
    #     dummy_array = np.random.randn(100, 100, 100)
    #     mock_zarr_root.__getitem__.return_value = dummy_array
    #     mock_tomogram.zarr.return_value = "dummy_zarr_path"
    #
    #     # Create the dataset
    #     dataset = MinimalCopickDataset(
    #         dataset_id=10440,
    #         overlay_root="/tmp/test/",
    #         boxsize=(32, 32, 32),
    #         voxel_spacing=10.0,
    #         include_background=True,
    #         background_ratio=0.2,
    #     )
    #
    #     # Test the dataset properties
    #     self.assertIsNotNone(dataset)
    #     self.assertEqual(dataset.dataset_id, 10440)
    #     self.assertEqual(dataset.boxsize, (32, 32, 32))
    #     self.assertEqual(dataset.voxel_spacing, 10.0)
    #     self.assertTrue(dataset.include_background)
    #
    #     # Check the class names
    #     expected_class_names = ["object1", "object2"]
    #     if dataset.include_background:
    #         expected_class_names.append("background")
    #
    #     self.assertEqual(set(dataset.keys()), set(expected_class_names))
    #
    #     # Test length - with 5 points per object (2 objects) and background_ratio of 0.2,
    #     # we expect 10 object points + approximately 2 background points
    #     # However, the exact number of background points may vary due to random sampling
    #     # So we just check it's at least the total object points
    #     self.assertGreaterEqual(len(dataset), 10)
    #
    #     # Test getting an item
    #     volume, label = dataset[0]
    #
    #     # Check the shape and type
    #     self.assertEqual(volume.shape, (1, 32, 32, 32))  # [C, D, H, W]
    #     self.assertIsInstance(volume, torch.Tensor)
    #     self.assertIsInstance(label, int)
    #
    #     # Check label is in the expected range
    #     self.assertIn(label, [-1, 0, 1])  # -1 for background, 0-1 for objects
    #
    #     # Test the class distribution
    #     distribution = dataset.get_class_distribution()
    #
    #     # Check each object has 5 points
    #     for obj_name in expected_class_names[:2]:  # Skip "background"
    #         self.assertEqual(distribution.get(obj_name, 0), 5)
    #
    #     # Test sample weights
    #     weights = dataset.get_sample_weights()
    #     self.assertEqual(len(weights), len(dataset))

    # @patch("zarr.open")
    # @patch("copick.from_czcdp_datasets")
    # def test_class_to_label_consistency(self, mock_from_czcdp, mock_zarr_open):
    #     """Test that class names and labels are consistent."""
    #     # Set up mocks (simplified version compared to above)
    #     mock_copick_root = MagicMock()
    #     mock_run = MagicMock()
    #     mock_vs = MagicMock()
    #     mock_tomogram = MagicMock()
    #
    #     # Configure mocks
    #     mock_from_czcdp.return_value = mock_copick_root
    #     mock_copick_root.runs = [mock_run]
    #     mock_copick_root.pickable_objects = [
    #         MagicMock(name="object1"),
    #         MagicMock(name="object2"),
    #         MagicMock(name="object3"),
    #     ]
    #
    #     for po in mock_copick_root.pickable_objects:
    #         po.name = po.name
    #
    #     mock_run.name = "test_run"
    #     mock_run.get_voxel_spacing.return_value = mock_vs
    #     mock_vs.tomograms = [mock_tomogram]
    #     mock_tomogram.tomo_type = "wbp-denoised"
    #
    #     # Setup picks - just one point per object for simplicity
    #     mock_picks_list = []
    #     for i, po in enumerate(mock_copick_root.pickable_objects):
    #         mock_pick = MagicMock()
    #         mock_pick.from_tool = True
    #         mock_pick.pickable_object_name = po.name
    #         mock_pick.numpy.return_value = (np.array([[100 * (i + 1), 100 * (i + 1), 100 * (i + 1)]]), None)
    #         mock_picks_list.append(mock_pick)
    #
    #     mock_run.get_picks.return_value = mock_picks_list
    #
    #     # Mock zarr.open to return a dummy array
    #     mock_zarr_root = MagicMock()
    #     mock_zarr_open.return_value = mock_zarr_root
    #     mock_zarr_root.__getitem__.return_value = np.zeros((500, 500, 500))
    #     mock_tomogram.zarr.return_value = "dummy_zarr_path"
    #
    #     # Create the dataset without background for simpler testing
    #     dataset = MinimalCopickDataset(
    #         dataset_id=10440,
    #         overlay_root="/tmp/test/",
    #         boxsize=(32, 32, 32),
    #         voxel_spacing=10.0,
    #         include_background=False,  # No background samples
    #     )
    #
    #     # Check length - should be exactly 3 points (one per object)
    #     self.assertEqual(len(dataset), 3)
    #
    #     # Verify class names and indices
    #     class_names = dataset.keys()
    #     self.assertEqual(len(class_names), 3)
    #
    #     # For each sample, verify the label matches the expected class
    #     for i in range(len(dataset)):
    #         _, label = dataset[i]
    #         # The label should be a valid index in the class_names list
    #         self.assertTrue(0 <= label < len(class_names))
    #         # Get the class name from the label
    #         class_name = class_names[label]
    #         # Make sure it's one of our expected class names
    #         self.assertIn(class_name, ["object1", "object2", "object3"])
    #
    #     # Test that the class distribution is correct
    #     distribution = dataset.get_class_distribution()
    #
    #     for name in ["object1", "object2", "object3"]:
    #         self.assertEqual(distribution.get(name, 0), 1)


if __name__ == "__main__":
    unittest.main()
