import os
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from copick_torch.detectors.dog_detector import DoGParticleDetector

class TestDoGDetector(unittest.TestCase):
    def setUp(self):
        # Create a simple 3D test volume
        self.volume = np.zeros((64, 64, 64), dtype=np.float32)
        
        # Add some particle-like features
        self.particle_positions = [
            (20, 20, 20),
            (40, 40, 40),
            (20, 40, 20),
            (40, 20, 40)
        ]
        
        # Create Gaussian-like particles at specified positions
        for pos in self.particle_positions:
            z, y, x = pos
            # Create a small Gaussian blob
            z_grid, y_grid, x_grid = np.mgrid[z-5:z+5, y-5:y+5, x-5:x+5]
            dist_sq = (z_grid - z)**2 + (y_grid - y)**2 + (x_grid - x)**2
            # Add Gaussian blob to the volume
            self.volume[z-5:z+5, y-5:y+5, x-5:x+5] += np.exp(-dist_sq / 8.0)
        
        # Initialize the detector
        self.detector = DoGParticleDetector(
            sigma1=1.0,
            sigma2=3.0,
            threshold_abs=0.1,
            min_distance=5
        )
    
    def test_init(self):
        """Test detector initialization."""
        self.assertEqual(self.detector.sigma1, 1.0)
        self.assertEqual(self.detector.sigma2, 3.0)
        self.assertEqual(self.detector.threshold_abs, 0.1)
        self.assertEqual(self.detector.min_distance, 5)
    
    def test_detect(self):
        """Test basic particle detection."""
        # Detect particles
        peaks = self.detector.detect(self.volume)
        
        # Check if we get some peaks
        self.assertGreater(len(peaks), 0)
        
        # Check that peaks array has correct shape
        self.assertEqual(peaks.shape[1], 3)  # Each peak should have (z, y, x) coordinates
    
    def test_detect_with_scores(self):
        """Test particle detection with scores."""
        # Detect particles and get scores
        peaks, scores = self.detector.detect(self.volume, return_scores=True)
        
        # Check if we get some peaks
        self.assertGreater(len(peaks), 0)
        
        # Check that peaks and scores have same length
        self.assertEqual(len(peaks), len(scores))
    
    def test_detect_multiscale(self):
        """Test multiscale particle detection."""
        # Define multiple scales
        sigma_pairs = [(1.0, 2.0), (2.0, 4.0)]
        
        # Detect particles at multiple scales
        peaks = self.detector.detect_multiscale(self.volume, sigma_pairs)
        
        # Check if we get some peaks
        self.assertGreater(len(peaks), 0)
    
    def test_optimize_parameters(self):
        """Test parameter optimization."""
        # Create a mock optimize function to avoid long computation
        with patch.object(self.detector, '_calculate_metrics', return_value=(0.8, 0.7, 0.75)):
            # Run optimization with reduced parameter space
            result = self.detector.optimize_parameters(
                self.volume,
                np.array(self.particle_positions),
                sigma1_range=(1.0, 1.5, 0.5),
                sigma2_range=(2.0, 2.5, 0.5),
                threshold_range=(0.1, 0.2, 0.1),
                min_distance_range=(5, 6, 1)
            )
            
            # Check if we get a result
            self.assertIsInstance(result, dict)
            self.assertIn('f1', result)
    
    def test_metrics_calculation(self):
        """Test metrics calculation."""
        # Create detected peaks close to ground truth
        detected = np.array([
            [21, 21, 21],  # Close to first particle
            [41, 41, 41],  # Close to second particle
            [30, 30, 30]   # False positive
        ])
        
        ground_truth = np.array(self.particle_positions)
        
        # Calculate metrics
        precision, recall, f1 = self.detector._calculate_metrics(
            detected, ground_truth, tolerance=5.0
        )
        
        # Check reasonable metrics
        self.assertGreaterEqual(precision, 0.0)
        self.assertLessEqual(precision, 1.0)
        self.assertGreaterEqual(recall, 0.0)
        self.assertLessEqual(recall, 1.0)
        self.assertGreaterEqual(f1, 0.0)
        self.assertLessEqual(f1, 1.0)

if __name__ == '__main__':
    unittest.main()
