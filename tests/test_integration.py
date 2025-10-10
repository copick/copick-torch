import os
import unittest

import pytest
import torch
from torch.utils.data import DataLoader

from copick_torch.copick import CopickDataset

# This test requires actual data
# Skip it if the test config file is not available
TEST_CONFIG_PATH = os.environ.get("COPICK_TEST_CONFIG", "./examples/czii_object_detection_training.json")


@pytest.mark.skipif(not os.path.exists(TEST_CONFIG_PATH), reason="Test config file not available")
class TestIntegration(unittest.TestCase):
    """Integration tests that require actual data.

    These tests will be skipped if the test data is not available.
    """

    def setUp(self):
        # Create a cache directory if it doesn't exist
        self.cache_dir = os.path.join(os.path.dirname(__file__), "test_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def test_dataset_with_real_data(self):
        """Test dataset with real data if available."""
        # Skip if data not available
        if not os.path.exists(TEST_CONFIG_PATH):
            self.skipTest("Test config file not available")

        # Initialize dataset with small boxsize and limited samples for faster testing
        dataset = CopickDataset(
            config_path=TEST_CONFIG_PATH,
            boxsize=(16, 16, 16),
            augment=False,
            cache_dir=self.cache_dir,
            max_samples=5,
        )

        # If no data was loaded (possibly because tomograms couldn't be accessed),
        # skip the test instead of failing
        if len(dataset) == 0:
            self.skipTest("No data could be loaded from the test configuration")

        # Test accessing an item only if dataset is not empty
        if len(dataset) > 0:
            volume, label = dataset[0]
            self.assertIsInstance(volume, torch.Tensor)
            self.assertEqual(volume.shape[0], 1)  # Check channel dimension

            # Test dataloader
            dataloader = DataLoader(dataset, batch_size=2)
            batch = next(iter(dataloader))
            self.assertEqual(len(batch), 2)  # Input and label
            self.assertEqual(batch[0].shape[0], min(2, len(dataset)))  # Batch dimension

    def test_examples_method(self):
        """Test the examples method with real data if available."""
        # Skip if data not available
        if not os.path.exists(TEST_CONFIG_PATH):
            self.skipTest("Test config file not available")

        # Initialize dataset
        dataset = CopickDataset(
            config_path=TEST_CONFIG_PATH,
            boxsize=(16, 16, 16),
            augment=False,
            cache_dir=self.cache_dir,
            max_samples=5,
        )

        # If no data was loaded, skip the test
        if len(dataset) == 0:
            self.skipTest("No data could be loaded from the test configuration")

        # Get examples
        examples, class_names = dataset.examples()

        # Check that we got examples for each class
        if examples is not None:
            self.assertEqual(len(examples), len(class_names))
            self.assertIsInstance(examples, torch.Tensor)
            self.assertEqual(examples.shape[1], 1)  # Channel dimension
            self.assertEqual(examples.shape[2:], (16, 16, 16))  # Spatial dimensions


if __name__ == "__main__":
    unittest.main()
