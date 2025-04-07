import os
import unittest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from copick_torch.detectors.monai_detector import MONAIParticleDetector

class TestMONAIDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            cls.device = "cpu"
        else:
            cls.device = "cuda"
    
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
    
    def test_init(self):
        """Test detector initialization with minimal configuration."""
        try:
            detector = MONAIParticleDetector(
                spatial_dims=3,
                num_classes=1,
                device="cpu" # Use CPU for testing
            )
            self.assertIsNotNone(detector)
        except Exception as e:
            self.fail(f"Detector initialization failed with exception: {e}")
    
    def test_detector_attributes(self):
        """Test that detector has the expected attributes."""
        detector = MONAIParticleDetector(
            spatial_dims=3,
            num_classes=1,
            device="cpu"
        )
        
        # Check attributes
        self.assertEqual(detector.spatial_dims, 3)
        self.assertEqual(detector.num_classes, 1)
        self.assertEqual(detector.device, "cpu")
        
        # Check that MONAI detector is created
        self.assertIsNotNone(detector.detector)
    
    @unittest.skipIf(torch.cuda.is_available() == False, "CUDA not available")
    def test_cuda_detection(self):
        """Test detection with CUDA if available."""
        detector = MONAIParticleDetector(
            spatial_dims=3,
            num_classes=1,
            device="cuda",
            detection_threshold=0.1  # Lower threshold for test
        )
        
        # Mock the detector forward method to return predefined detections
        mock_detections = [{
            "boxes": torch.tensor([[15, 15, 15, 25, 25, 25], 
                                   [35, 35, 35, 45, 45, 45]]).cuda(),
            "labels": torch.tensor([0, 0]).cuda(),
            "labels_scores": torch.tensor([0.9, 0.8]).cuda()
        }]
        with patch.object(detector.detector, 'forward', return_value=mock_detections):
            # Convert volume to tensor
            volume_tensor = torch.from_numpy(self.volume).unsqueeze(0).cuda()  # Add channel dimension
            
            # Detect particles
            coords, scores = detector.detect(volume_tensor, return_scores=True)
            
            # Check shape and content
            self.assertEqual(coords.shape, (2, 3))  # 2 particles, 3D coordinates
            self.assertEqual(scores.shape, (2,))  # 2 confidence scores
            
            # Check coordinate calculation
            np.testing.assert_allclose(coords[0], [20, 20, 20], atol=1.0)  # Center of first box
            np.testing.assert_allclose(coords[1], [40, 40, 40], atol=1.0)  # Center of second box
    
    def test_cpu_detection(self):
        """Test detection on CPU."""
        detector = MONAIParticleDetector(
            spatial_dims=3,
            num_classes=1,
            device="cpu",
            detection_threshold=0.1  # Lower threshold for test
        )
        
        # Mock the detector forward method to return predefined detections
        mock_detections = [{
            "boxes": torch.tensor([[15, 15, 15, 25, 25, 25], 
                                   [35, 35, 35, 45, 45, 45]]),
            "labels": torch.tensor([0, 0]),
            "labels_scores": torch.tensor([0.9, 0.8])
        }]
        with patch.object(detector.detector, 'forward', return_value=mock_detections):
            # Detect particles
            coords = detector.detect(self.volume)
            
            # Check shape
            self.assertEqual(coords.shape, (2, 3))  # 2 particles, 3D coordinates
            
            # Check coordinate calculation
            np.testing.assert_allclose(coords[0], [20, 20, 20], atol=1.0)  # Center of first box
            np.testing.assert_allclose(coords[1], [40, 40, 40], atol=1.0)  # Center of second box
    
    def test_2d_detection(self):
        """Test 2D detection."""
        # Create a 2D test image
        image = np.zeros((64, 64), dtype=np.float32)
        
        # Add some particle-like features
        particle_positions_2d = [(20, 20), (40, 40)]
        
        for pos in particle_positions_2d:
            y, x = pos
            # Create a small Gaussian blob
            y_grid, x_grid = np.mgrid[y-5:y+5, x-5:x+5]
            dist_sq = (y_grid - y)**2 + (x_grid - x)**2
            # Add Gaussian blob to the image
            image[y-5:y+5, x-5:x+5] += np.exp(-dist_sq / 8.0)
            
        # Create 2D detector
        detector = MONAIParticleDetector(
            spatial_dims=2,
            num_classes=1,
            device="cpu",
            detection_threshold=0.1  # Lower threshold for test
        )
        
        # Mock the detector forward method to return predefined detections
        mock_detections = [{
            "boxes": torch.tensor([[15, 15, 25, 25], 
                                  [35, 35, 45, 45]]),
            "labels": torch.tensor([0, 0]),
            "labels_scores": torch.tensor([0.9, 0.8])
        }]
        with patch.object(detector.detector, 'forward', return_value=mock_detections):
            # Detect particles
            coords = detector.detect(image)
            
            # Check shape
            self.assertEqual(coords.shape, (2, 2))  # 2 particles, 2D coordinates
            
            # Check coordinate calculation
            np.testing.assert_allclose(coords[0], [20, 20], atol=1.0)  # Center of first box
            np.testing.assert_allclose(coords[1], [40, 40], atol=1.0)  # Center of second box
    
    def test_save_load_weights(self):
        """Test saving and loading weights."""
        import tempfile
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            # Create detector
            detector = MONAIParticleDetector(
                spatial_dims=3,
                num_classes=1,
                device="cpu"
            )
            
            # Save weights
            detector.save_weights(tmp.name)
            
            # Check that file exists and has content
            self.assertTrue(os.path.exists(tmp.name))
            self.assertGreater(os.path.getsize(tmp.name), 0)
            
            # Create a new detector
            detector2 = MONAIParticleDetector(
                spatial_dims=3,
                num_classes=1,
                device="cpu"
            )
            
            # Load weights
            detector2.load_weights(tmp.name)
            
            # Check that weights are loaded (this is a basic check, not comprehensive)
            # Here we just check that loading doesn't raise an exception
            self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
