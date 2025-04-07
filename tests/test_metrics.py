import unittest
import numpy as np

from copick_torch.metrics import (
    calculate_distances,
    calculate_precision_recall_f1,
    calculate_average_precision,
    calculate_detector_metrics
)

class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Create test data
        self.detected = np.array([
            [10, 10, 10],
            [30, 30, 30],
            [50, 50, 50],
            [70, 70, 70]
        ])
        
        self.ground_truth = np.array([
            [12, 12, 12],  # Close to first detection
            [32, 32, 32],  # Close to second detection
            [90, 90, 90]   # No matching detection
        ])
        
        self.scores = np.array([0.9, 0.8, 0.7, 0.6])
    
    def test_calculate_distances(self):
        """Test distance calculation between detected and ground truth particles."""
        distances = calculate_distances(self.detected, self.ground_truth)
        
        # Check shape
        self.assertEqual(distances.shape, (4, 3))
        
        # Check a few specific distances
        # Distance between [10,10,10] and [12,12,12] should be sqrt(12)
        self.assertAlmostEqual(distances[0, 0], np.sqrt(12), delta=1e-6)
        
        # Distance between [30,30,30] and [32,32,32] should be sqrt(12)
        self.assertAlmostEqual(distances[1, 1], np.sqrt(12), delta=1e-6)
        
        # Distance between [70,70,70] and [90,90,90] should be sqrt(1200)
        self.assertAlmostEqual(distances[3, 2], np.sqrt(1200), delta=1e-6)
    
    def test_precision_recall_f1(self):
        """Test precision, recall, and F1 calculation."""
        # With tolerance 5.0, only the first two detections should match
        precision, recall, f1 = calculate_precision_recall_f1(
            self.detected, self.ground_truth, tolerance=5.0
        )
        
        # Expected results
        # Precision = 2/4 = 0.5 (2 true positives out of 4 detections)
        # Recall = 2/3 = 0.667 (2 true positives out of 3 ground truth)
        # F1 = 2 * 0.5 * 0.667 / (0.5 + 0.667) = 0.572
        self.assertAlmostEqual(precision, 0.5, delta=1e-6)
        self.assertAlmostEqual(recall, 2/3, delta=1e-6)
        self.assertAlmostEqual(f1, 2 * 0.5 * (2/3) / (0.5 + 2/3), delta=1e-6)
        
        # With tolerance 50.0, three detections should match (leaving one ground truth unmatched)
        precision, recall, f1 = calculate_precision_recall_f1(
            self.detected, self.ground_truth, tolerance=50.0
        )
        
        # Expected results
        # Precision = 3/4 = 0.75 (3 true positives out of 4 detections)
        # Recall = 3/3 = 1.0 (3 true positives out of 3 ground truth)
        # F1 = 2 * 0.75 * 1.0 / (0.75 + 1.0) = 0.857
        self.assertAlmostEqual(precision, 0.75, delta=1e-6)
        self.assertAlmostEqual(recall, 1.0, delta=1e-6)
        self.assertAlmostEqual(f1, 2 * 0.75 * 1.0 / (0.75 + 1.0), delta=1e-6)
    
    def test_average_precision(self):
        """Test average precision calculation."""
        # With tolerance 5.0, only the first two detections should match
        ap, precision_values, recall_values, thresholds = calculate_average_precision(
            self.detected, self.scores, self.ground_truth, tolerance=5.0
        )
        
        # Check that values are returned
        self.assertTrue(isinstance(ap, float))
        self.assertTrue(isinstance(precision_values, list))
        self.assertTrue(isinstance(recall_values, list))
        self.assertTrue(isinstance(thresholds, list))
        
        # Check that values are in correct range
        self.assertGreaterEqual(ap, 0.0)
        self.assertLessEqual(ap, 1.0)
        for p in precision_values:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)
        for r in recall_values:
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 1.0)
    
    def test_detector_metrics(self):
        """Test comprehensive detector metrics calculation."""
        # Calculate metrics
        metrics = calculate_detector_metrics(
            self.detected, self.ground_truth, self.scores, tolerance=5.0
        )
        
        # Check that expected keys are present
        expected_keys = [
            "precision", "recall", "f1_score", 
            "num_detections", "num_ground_truth",
            "true_positives", "false_positives", "false_negatives",
            "average_precision", "precision_values", "recall_values", "thresholds"
        ]
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Check specific values
        self.assertEqual(metrics["num_detections"], 4)
        self.assertEqual(metrics["num_ground_truth"], 3)
        self.assertEqual(metrics["true_positives"], 2)
        self.assertEqual(metrics["false_positives"], 2)
        self.assertEqual(metrics["false_negatives"], 1)
    
    def test_empty_detections(self):
        """Test metrics with empty detections."""
        empty_detected = np.zeros((0, 3))
        
        # Calculate metrics
        precision, recall, f1 = calculate_precision_recall_f1(
            empty_detected, self.ground_truth, tolerance=5.0
        )
        
        # With no detections, precision is undefined (set to 0), recall is 0
        self.assertEqual(precision, 0.0)
        self.assertEqual(recall, 0.0)
        self.assertEqual(f1, 0.0)
    
    def test_empty_ground_truth(self):
        """Test metrics with empty ground truth."""
        empty_ground_truth = np.zeros((0, 3))
        
        # Calculate metrics
        precision, recall, f1 = calculate_precision_recall_f1(
            self.detected, empty_ground_truth, tolerance=5.0
        )
        
        # With no ground truth, precision is 0, recall is undefined (set to 0)
        self.assertEqual(precision, 0.0)
        self.assertEqual(recall, 0.0)
        self.assertEqual(f1, 0.0)
    
    def test_detector_metrics_no_scores(self):
        """Test metrics calculation without confidence scores."""
        # Calculate metrics without scores
        metrics = calculate_detector_metrics(
            self.detected, self.ground_truth, tolerance=5.0
        )
        
        # Check that AP-related keys are not present
        self.assertNotIn("average_precision", metrics)
        self.assertNotIn("precision_values", metrics)
        self.assertNotIn("recall_values", metrics)
        self.assertNotIn("thresholds", metrics)
        
        # Check that other metrics are still calculated
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)

if __name__ == '__main__':
    unittest.main()
