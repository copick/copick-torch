import unittest
from collections import Counter

import numpy as np
import torch

from copick_torch.samplers import ClassBalancedSampler


class TestClassBalancedSampler(unittest.TestCase):
    def setUp(self):
        # Create unbalanced data for testing
        self.labels = [0, 0, 0, 0, 1, 1, 2]  # Class 0 is overrepresented

    def test_init(self):
        """Test sampler initialization and weight calculation."""
        sampler = ClassBalancedSampler(self.labels)

        # Check initialization
        np.testing.assert_array_equal(sampler.labels, np.array(self.labels))
        self.assertEqual(sampler.num_samples, len(self.labels))
        self.assertTrue(sampler.replacement)

        # Check class counts
        expected_counts = {0: 4, 1: 2, 2: 1}
        self.assertEqual(sampler.class_counts, expected_counts)

        # Check weights calculation
        # Weight per class should be inversely proportional to count
        expected_weights = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 2, 1 / 2, 1])  # Class 0  # Class 1  # Class 2

        # Normalize to sum to 1
        expected_weights = expected_weights / expected_weights.sum()

        # Check weights are approximately equal (account for floating point precision)
        np.testing.assert_allclose(sampler.weights, expected_weights)

    def test_init_with_custom_samples(self):
        """Test initialization with custom number of samples."""
        custom_samples = 20
        sampler = ClassBalancedSampler(self.labels, num_samples=custom_samples)

        self.assertEqual(sampler.num_samples, custom_samples)
        self.assertEqual(len(sampler), custom_samples)

    def test_init_with_no_replacement(self):
        """Test initialization with replacement=False."""
        sampler = ClassBalancedSampler(self.labels, replacement=False)

        self.assertFalse(sampler.replacement)

    def test_iter(self):
        """Test __iter__ method produces indices within range."""
        # Set a fixed seed for reproducibility
        np.random.seed(42)

        # Create sampler
        sampler = ClassBalancedSampler(self.labels, num_samples=50)

        # Get indices
        indices = list(iter(sampler))

        # Check length
        self.assertEqual(len(indices), 50)

        # Check all indices are within valid range
        self.assertTrue(all(0 <= idx < len(self.labels) for idx in indices))

        # Count class occurrences to verify balance
        sampled_labels = [self.labels[idx] for idx in indices]
        label_counts = Counter(sampled_labels)

        # Check that the minority class (2) has more representation than its original proportion
        original_proportions = {0: 4 / 7, 1: 2 / 7, 2: 1 / 7}  # ~57%  # ~29%  # ~14%

        sampled_proportions = {cls: count / 50 for cls, count in label_counts.items()}

        # The resampled proportion for class 2 should be higher than its original proportion
        self.assertGreater(sampled_proportions[2], original_proportions[2])

        # Similarly, the overrepresented class 0 should have a lower proportion
        self.assertLess(sampled_proportions[0], original_proportions[0])

    def test_len(self):
        """Test __len__ method returns correct number of samples."""
        sampler = ClassBalancedSampler(self.labels)
        self.assertEqual(len(sampler), len(self.labels))

        # Test with custom number of samples
        custom_samples = 20
        sampler = ClassBalancedSampler(self.labels, num_samples=custom_samples)
        self.assertEqual(len(sampler), custom_samples)


if __name__ == "__main__":
    unittest.main()
