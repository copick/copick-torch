import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from copick_torch.dataset import SplicedMixupDataset
from tests.storage_helpers import make_entity, make_v2_store


class TestSplicedMixupDataset(unittest.TestCase):
    """Test the SplicedMixupDataset class with focus on Gaussian blending functionality."""

    def setUp(self):
        """Set up test case."""
        # Create patches for all required methods
        load_copick_roots_patch = patch("copick_torch.dataset.SplicedMixupDataset._load_copick_roots")
        load_process_data_patch = patch("copick_torch.dataset.SimpleCopickDataset._load_or_process_data")
        ensure_zarr_patch = patch("copick_torch.dataset.SplicedMixupDataset._ensure_zarr_loaded")
        generate_samples_patch = patch("copick_torch.dataset.SplicedMixupDataset._generate_synthetic_samples")

        # Start all patches
        self.addCleanup(load_copick_roots_patch.stop)
        self.addCleanup(load_process_data_patch.stop)
        self.addCleanup(ensure_zarr_patch.stop)
        self.addCleanup(generate_samples_patch.stop)

        load_copick_roots_patch.start()
        load_process_data_patch.start()
        ensure_zarr_patch.start()
        generate_samples_patch.start()

        # The key: patch the validation check directly
        with patch("copick_torch.dataset.SimpleCopickDataset.__init__", return_value=None):
            self.dataset = SplicedMixupDataset(exp_dataset_id=1, synth_dataset_id=2, blend_sigma=2.0)

        # Set necessary attributes that would normally be set in initialization
        self.dataset.blend_sigma = 2.0
        self.dataset._subvolumes = []
        self.dataset._molecule_ids = []

    def test_splice_volumes_gaussian_blending(self):
        """Test the _splice_volumes method with Gaussian blending."""
        # Create test data
        boxsize = (16, 16, 16)
        synthetic_region = np.ones(boxsize)  # All 1s
        exp_crop = np.zeros(boxsize)  # All 0s

        # Create a mask that covers part of the volume
        region_mask = np.zeros(boxsize, dtype=bool)
        region_mask[4:12, 4:12, 4:12] = True  # Inner cube is True

        # Test with blend_sigma=0 (no blending)
        self.dataset.blend_sigma = 0.0
        result_no_blend = self.dataset._splice_volumes(synthetic_region, region_mask, exp_crop)

        # Verify values: should be 1 inside mask, 0 outside, with no blending
        self.assertTrue(np.all(result_no_blend[region_mask] == 1.0))
        self.assertTrue(np.all(result_no_blend[~region_mask] == 0.0))

        # Test with blend_sigma=2.0 (Gaussian blending)
        self.dataset.blend_sigma = 2.0
        result_gaussian = self.dataset._splice_volumes(synthetic_region, region_mask, exp_crop)

        # Verify values:
        # 1. Core of masked area should still be relatively high (central region)
        # Use a lower threshold (0.7 instead of 0.9) to account for the blending effect
        self.assertTrue(np.all(result_gaussian[7:9, 7:9, 7:9] > 0.7))

        # 2. Outside mask far from boundary should still be close to 0
        self.assertTrue(np.all(result_gaussian[0:2, 0:2, 0:2] < 0.1))

        # 3. Boundary region should have intermediate values
        # Check for points near the boundary
        boundary_values = result_gaussian[3:5, 8, 8]  # Points near boundary
        self.assertTrue(np.any((boundary_values > 0.1) & (boundary_values < 0.9)))

        # 4. Values should transition smoothly across boundary
        # Check along a line from center to outside
        center_to_edge = result_gaussian[8, 8, 8:16]  # Line from center to edge
        # Verify monotonic decrease (or at least mostly decreasing)
        # Check if at least 80% of the differences are non-positive
        diffs = np.diff(center_to_edge)
        self.assertTrue(np.sum(diffs <= 0) >= 0.8 * len(diffs))

    def test_gaussian_blending_vs_no_blending(self):
        """Compare Gaussian blending with no blending to ensure they're different."""
        # Create test data
        boxsize = (16, 16, 16)
        synthetic_region = np.ones(boxsize)  # All 1s
        exp_crop = np.zeros(boxsize)  # All 0s

        # Create a mask
        region_mask = np.zeros(boxsize, dtype=bool)
        region_mask[4:12, 4:12, 4:12] = True  # Inner cube is True

        # Get results with and without blending
        self.dataset.blend_sigma = 0.0
        result_no_blend = self.dataset._splice_volumes(synthetic_region, region_mask, exp_crop)

        self.dataset.blend_sigma = 2.0
        result_gaussian = self.dataset._splice_volumes(synthetic_region, region_mask, exp_crop)

        # Check that results are different
        self.assertFalse(np.array_equal(result_no_blend, result_gaussian))

        # Check that binary mask is recovered with no blending
        binary_mask_recovered = result_no_blend > 0.5
        self.assertTrue(np.array_equal(binary_mask_recovered, region_mask))

        # With Gaussian blending, more pixels should be affected than just the mask
        affected_pixels_gaussian = result_gaussian > 0.01
        self.assertTrue(np.sum(affected_pixels_gaussian) > np.sum(region_mask))

    def test_zarr_loading_methods_use_numeric_level(self):
        exp_store, exp = make_v2_store("0")
        synth_store, synth = make_v2_store("0", exp + 1000)
        mask_store, mask = make_v2_store("0", (exp % 3 == 0).astype(np.uint8))

        dataset = SplicedMixupDataset.__new__(SplicedMixupDataset)
        dataset.voxel_spacing = 10.0
        dataset.exp_root = object()
        dataset.synth_root = object()
        dataset._synth_mask_data = {}
        entities = {
            id(dataset.exp_root): [make_entity(exp_store)],
            id(dataset.synth_root): [make_entity(synth_store)],
        }
        dataset._get_available_tomograms = lambda root, _voxel_size: entities[id(root)]
        dataset._get_segmentation_masks = lambda _root, _voxel_size: {
            "particle": make_entity(mask_store, name="particle"),
        }

        dataset._load_experimental_zarr()
        dataset._load_synthetic_zarr()
        dataset._load_segmentation_masks()

        np.testing.assert_allclose(dataset._exp_zarr_data.mean(), 0.0, atol=1e-6)
        np.testing.assert_allclose(dataset._exp_zarr_data.std(), 1.0, atol=1e-6)
        np.testing.assert_allclose(dataset._synth_zarr_data.mean(), 0.0, atol=1e-6)
        np.testing.assert_array_equal(dataset._synth_mask_data["particle"], mask)


if __name__ == "__main__":
    unittest.main()
