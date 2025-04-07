"""
Difference of Gaussian (DoG) particle detector for CryoET data.

This module implements a simple but effective particle detector based on the Difference of Gaussian (DoG)
method, which is widely used in particle picking for cryo-EM and cryo-ET data.
"""

import logging
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from typing import List, Tuple, Union, Optional, Dict, Any, Sequence

class DoGParticleDetector:
    """
    Difference of Gaussian (DoG) particle detector for CryoET data.
    
    This detector applies Gaussian filters with two different sigma values to the input volume,
    then subtracts the more blurred volume from the less blurred one to enhance particle-like features.
    Local maxima in the resulting volume are identified as potential particle locations.
    
    Args:
        sigma1: sigma value for the first Gaussian filter (smaller)
        sigma2: sigma value for the second Gaussian filter (larger)
        threshold_abs: absolute threshold for peak detection
        min_distance: minimum distance between peaks (in voxels)
        exclude_border: exclude border region of this size (in voxels)
        normalize: whether to normalize the input volume before processing
        invert: whether to invert the contrast (for dark particles on light background)
        prefilter: optional filter to apply before DoG (e.g., 'median', 'gaussian')
        prefilter_size: size parameter for prefilter
    """
    
    def __init__(
        self,
        sigma1: float = 1.0,
        sigma2: float = 3.0,
        threshold_abs: float = 0.1,
        min_distance: int = 5,
        exclude_border: int = 2,
        normalize: bool = True,
        invert: bool = False,
        prefilter: Optional[str] = None,
        prefilter_size: float = 1.0
    ):
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.threshold_abs = threshold_abs
        self.min_distance = min_distance
        self.exclude_border = exclude_border
        self.normalize = normalize
        self.invert = invert
        self.prefilter = prefilter
        self.prefilter_size = prefilter_size
        
        self.logger = logging.getLogger(__name__)
    
    def detect(self, volume: np.ndarray, return_scores: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Detect particles in a 3D volume using Difference of Gaussian method.
        
        Args:
            volume: input 3D volume
            return_scores: whether to return the peak values at each detected location
            
        Returns:
            np.ndarray: particle coordinates (N, 3)
            np.ndarray (optional): peak values at each location (N,)
        """
        # Check input dimensions
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {volume.shape}")
        
        # Make a copy to avoid modifying the input
        vol = volume.copy()
        
        # Invert contrast if needed (for dark particles on light background)
        if self.invert:
            vol = -vol
        
        # Normalize if requested
        if self.normalize:
            vol = (vol - np.mean(vol)) / np.std(vol)
        
        # Apply prefilter if requested
        if self.prefilter == 'median':
            self.logger.info(f"Applying median filter with size {self.prefilter_size}")
            vol = ndimage.median_filter(vol, size=self.prefilter_size)
        elif self.prefilter == 'gaussian':
            self.logger.info(f"Applying Gaussian filter with sigma {self.prefilter_size}")
            vol = ndimage.gaussian_filter(vol, sigma=self.prefilter_size)
        
        # Apply Gaussian filters
        self.logger.info(f"Applying DoG with sigma1={self.sigma1}, sigma2={self.sigma2}")
        vol_smooth1 = ndimage.gaussian_filter(vol, sigma=self.sigma1)
        vol_smooth2 = ndimage.gaussian_filter(vol, sigma=self.sigma2)
        
        # Calculate Difference of Gaussian
        dog_vol = vol_smooth1 - vol_smooth2
        
        # Find local maxima
        self.logger.info(f"Finding peaks with min_distance={self.min_distance}, threshold={self.threshold_abs}")
        peaks = peak_local_max(
            dog_vol,
            min_distance=self.min_distance,
            threshold_abs=self.threshold_abs,
            exclude_border=self.exclude_border,
            indices=True
        )
        
        # Return peak coordinates and peak values if requested
        if return_scores:
            peak_values = np.array([dog_vol[tuple(peak)] for peak in peaks])
            return peaks, peak_values
        
        return peaks
    
    def detect_multiscale(
        self, 
        volume: np.ndarray, 
        sigma_pairs: List[Tuple[float, float]],
        return_scores: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Detect particles at multiple scales by applying DoG with different sigma pairs.
        
        Args:
            volume: input 3D volume
            sigma_pairs: list of (sigma1, sigma2) pairs to try
            return_scores: whether to return the peak values at each detected location
            
        Returns:
            np.ndarray: particle coordinates (N, 3)
            np.ndarray (optional): peak values at each location (N,)
        """
        all_peaks = []
        all_values = []
        
        # Keep the original settings
        orig_sigma1 = self.sigma1
        orig_sigma2 = self.sigma2
        
        # Try each sigma pair
        for sigma1, sigma2 in sigma_pairs:
            self.sigma1 = sigma1
            self.sigma2 = sigma2
            
            # Detect particles with this sigma pair
            if return_scores:
                peaks, values = self.detect(volume, return_scores=True)
                all_peaks.append(peaks)
                all_values.append(values)
            else:
                peaks = self.detect(volume, return_scores=False)
                all_peaks.append(peaks)
        
        # Restore original settings
        self.sigma1 = orig_sigma1
        self.sigma2 = orig_sigma2
        
        # Combine results
        if all_peaks:
            combined_peaks = np.vstack(all_peaks)
            
            if return_scores:
                combined_values = np.hstack(all_values)
                return combined_peaks, combined_values
            return combined_peaks
        
        # Return empty array if no peaks found
        if return_scores:
            return np.zeros((0, 3)), np.zeros(0)
        return np.zeros((0, 3))
    
    def optimize_parameters(
        self, 
        volume: np.ndarray, 
        ground_truth: np.ndarray, 
        sigma1_range: Tuple[float, float, float] = (0.5, 3.0, 0.5),
        sigma2_range: Tuple[float, float, float] = (1.0, 5.0, 0.5),
        threshold_range: Tuple[float, float, float] = (0.05, 0.5, 0.05),
        min_distance_range: Tuple[int, int, int] = (3, 10, 1),
        tolerance: float = 5.0
    ) -> Dict[str, Any]:
        """
        Optimize detector parameters by grid search against ground truth particles.
        
        Args:
            volume: input 3D volume
            ground_truth: array of ground truth particle coordinates (N, 3)
            sigma1_range: (start, stop, step) for sigma1 values to try
            sigma2_range: (start, stop, step) for sigma2 values to try
            threshold_range: (start, stop, step) for threshold values to try 
            min_distance_range: (start, stop, step) for min_distance values to try
            tolerance: maximum distance for a detected particle to be considered a match
            
        Returns:
            dict: optimal parameters found
        """
        best_params = {}
        best_f1 = 0.0
        
        # Create parameter grid
        sigma1_values = np.arange(sigma1_range[0], sigma1_range[1] + 1e-5, sigma1_range[2])
        sigma2_values = np.arange(sigma2_range[0], sigma2_range[1] + 1e-5, sigma2_range[2])
        threshold_values = np.arange(threshold_range[0], threshold_range[1] + 1e-5, threshold_range[2])
        min_distance_values = np.arange(
            min_distance_range[0], min_distance_range[1] + 1, min_distance_range[2], dtype=int
        )
        
        total_combinations = (
            len(sigma1_values) * len(sigma2_values) * 
            len(threshold_values) * len(min_distance_values)
        )
        self.logger.info(f"Grid search over {total_combinations} parameter combinations")
        
        # Iterate over all parameter combinations
        for sigma1 in sigma1_values:
            for sigma2 in sigma2_values:
                # Skip invalid combinations
                if sigma2 <= sigma1:
                    continue
                    
                for threshold in threshold_values:
                    for min_distance in min_distance_values:
                        # Update detector parameters
                        self.sigma1 = sigma1
                        self.sigma2 = sigma2
                        self.threshold_abs = threshold
                        self.min_distance = min_distance
                        
                        # Detect particles with current parameters
                        detected_peaks = self.detect(volume)
                        
                        # Calculate metrics
                        precision, recall, f1 = self._calculate_metrics(detected_peaks, ground_truth, tolerance)
                        
                        # Update best parameters if F1 score improved
                        if f1 > best_f1:
                            best_f1 = f1
                            best_params = {
                                'sigma1': sigma1,
                                'sigma2': sigma2,
                                'threshold_abs': threshold,
                                'min_distance': min_distance,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1
                            }
                            
                            self.logger.info(
                                f"New best: F1={f1:.3f}, P={precision:.3f}, R={recall:.3f} with "
                                f"sigma1={sigma1}, sigma2={sigma2}, thresh={threshold}, min_dist={min_distance}"
                            )
        
        # Set detector to best parameters
        self.sigma1 = best_params['sigma1']
        self.sigma2 = best_params['sigma2']
        self.threshold_abs = best_params['threshold_abs']
        self.min_distance = best_params['min_distance']
        
        return best_params
    
    def _calculate_metrics(
        self, 
        detected: np.ndarray, 
        ground_truth: np.ndarray, 
        tolerance: float
    ) -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F1 score for particle detection.
        
        Args:
            detected: detected particle coordinates
            ground_truth: ground truth particle coordinates
            tolerance: maximum distance to consider a detection as correct
            
        Returns:
            tuple: (precision, recall, f1_score)
        """
        if len(detected) == 0:
            return 0, 0, 0
            
        if len(ground_truth) == 0:
            return 0, 0, 0
        
        # Calculate distances between all detections and ground truth particles
        true_positives = 0
        matched_gt = set()
        
        # For each detected particle, find the closest ground truth
        for det in detected:
            # Calculate Euclidean distances to all ground truth particles
            distances = np.sqrt(np.sum((ground_truth - det)**2, axis=1))
            
            # Find the closest ground truth particle
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]
            
            # Consider it a match if the distance is within tolerance
            if min_dist <= tolerance and min_idx not in matched_gt:
                true_positives += 1
                matched_gt.add(min_idx)
        
        # Calculate metrics
        precision = true_positives / len(detected) if len(detected) > 0 else 0
        recall = true_positives / len(ground_truth) if len(ground_truth) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
