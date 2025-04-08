
import unittest
import torch
import numpy as np
from unittest.mock import patch
from copick_torch import MixupAugmentation


class TestMixupAugmentation(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        batch_size = 4
        channels = 1
        depth, height, width = 16, 16, 16
        
        # Create sample images and labels
        self.images = torch.randn(batch_size, channels, depth, height, width)
        self.labels = torch.tensor([0, 1, 2, 0])  # Sample class labels
        
    def test_init(self):
        """Test MixupAugmentation initialization with different alpha values."""
        # Test with default alpha
        mixup = MixupAugmentation()
        self.assertEqual(mixup.alpha, 0.2)
        
        # Test with custom alpha
        mixup = MixupAugmentation(alpha=0.5)
        self.assertEqual(mixup.alpha, 0.5)
        
    @patch('numpy.random.beta')
    @patch('torch.randperm')
    def test_call(self, mock_randperm, mock_beta):
        """Test the __call__ method of MixupAugmentation."""
        # Configure the mock to return a fixed value for beta
        mock_beta.return_value = 0.7
        
        # Configure randperm to return a fixed permutation that's guaranteed to be different
        # Create a permutation that swaps the first and second elements
        fixed_perm = torch.tensor([1, 0, 3, 2])
        mock_randperm.return_value = fixed_perm
        
        mixup = MixupAugmentation(alpha=0.2)
        
        # Call the mixup augmentation
        mixed_images, label_a, label_b, lam = mixup(self.images, self.labels)
        
        # Check output shapes
        self.assertEqual(mixed_images.shape, self.images.shape)
        self.assertEqual(label_a.shape, self.labels.shape)
        self.assertEqual(label_b.shape, self.labels.shape)
        
        # Verify lambda value
        self.assertEqual(lam, 0.7)
        
        # Verify the labels are correctly permuted
        self.assertTrue(torch.equal(label_b, self.labels[fixed_perm]))
        
        # Ensure the mocks were called with correct parameters
        mock_beta.assert_called_once_with(0.2, 0.2)
        mock_randperm.assert_called_once()
        
        # Verify mixing computation: mixed = lambda*original + (1-lambda)*permuted
        expected_mixed = 0.7 * self.images + 0.3 * self.images[fixed_perm]
        self.assertTrue(torch.allclose(mixed_images, expected_mixed))
            
    def test_call_with_zero_alpha(self):
        """Test mixup with alpha=0 which should return original images."""
        mixup = MixupAugmentation(alpha=0)
        
        # Call the mixup augmentation
        mixed_images, label_a, label_b, lam = mixup(self.images, self.labels)
        
        # Check that mixed_images are the same as original images
        self.assertEqual(lam, 1.0)
        self.assertTrue(torch.all(mixed_images == self.images))
        
    def test_mixup_criterion(self):
        """Test the mixup_criterion static method."""
        mixup = MixupAugmentation()
        
        # Define a simple criterion (MSE loss)
        criterion = torch.nn.MSELoss()
        
        # Create example predictions and labels
        pred = torch.randn(4, 3)  # 4 samples, 3 classes
        y_a = torch.tensor([0, 1, 2, 0])
        y_b = torch.tensor([1, 0, 1, 2])
        lam = 0.6
        
        # Convert to one-hot encoding for MSE loss
        y_a_one_hot = torch.nn.functional.one_hot(y_a, num_classes=3).float()
        y_b_one_hot = torch.nn.functional.one_hot(y_b, num_classes=3).float()
        
        # Calculate expected loss
        loss_a = criterion(pred, y_a_one_hot)
        loss_b = criterion(pred, y_b_one_hot)
        expected_loss = lam * loss_a + (1 - lam) * loss_b
        
        # Calculate loss using mixup_criterion
        mixed_loss = mixup.mixup_criterion(
            lambda p, y: criterion(p, torch.nn.functional.one_hot(y, num_classes=3).float()),
            pred, y_a, y_b, lam
        )
        
        # Verify the loss is calculated correctly
        self.assertAlmostEqual(mixed_loss.item(), expected_loss.item(), places=5)


if __name__ == '__main__':
    unittest.main()
