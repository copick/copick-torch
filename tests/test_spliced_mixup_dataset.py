import unittest
import os
import numpy as np
from unittest.mock import patch, MagicMock

from copick_torch import SplicedMixupDataset


class TestSplicedMixupDataset(unittest.TestCase):
    """Test the SplicedMixupDataset class with focus on Gaussian blending functionality."""

    def setUp(self):
        """Set up test case."""
        # Create a mock instance of SplicedMixupDataset for testing
        with patch('copick_torch.dataset.SplicedMixupDataset._load_copick_roots'):
            with patch('copick_torch.dataset.SimpleCopickDataset._load_or_process_data'):
                with patch('copick_torch.dataset.SplicedMixupDataset._ensure_zarr_loaded'):
                    with patch('copick_torch.dataset.SplicedMixupDataset._generate_synthetic_samples'):
                        self.dataset = SplicedMixupDataset(
                            exp_dataset_id=1,
                            synth_dataset_id=2,
                            blend_sigma=2.0
                        )

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
        # 1. Core of masked area should still be close to 1
        self.assertTrue(np.all(result_gaussian[6:10, 6:10, 6:10] > 0.9))
        
        # 2. Outside mask far from boundary should still be close to 0
        self.assertTrue(np.all(result_gaussian[0:2, 0:2, 0:2] < 0.1))
        
        # 3. Boundary region should have intermediate values
        # Check for points near the boundary
        boundary_values = result_gaussian[3:5, 8, 8]  # Points near boundary
        self.assertTrue(np.any((boundary_values > 0.1) & (boundary_values < 0.9)))
        
        # 4. Values should transition smoothly across boundary
        # Check along a line from center to outside
        center_to_edge = result_gaussian[8, 8, 8:16]  # Line from center to edge
        # Verify monotonic decrease
        self.assertTrue(np.all(np.diff(center_to_edge) <= 0))

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
        binary_mask_recovered = (result_no_blend > 0.5)
        self.assertTrue(np.array_equal(binary_mask_recovered, region_mask))
        
        # With Gaussian blending, more pixels should be affected than just the mask
        affected_pixels_gaussian = (result_gaussian > 0.01)
        self.assertTrue(np.sum(affected_pixels_gaussian) > np.sum(region_mask))


if __name__ == '__main__':
    unittest.main()
