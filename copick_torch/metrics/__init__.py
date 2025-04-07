"""
Evaluation metrics for particle detection performance.

This module provides various metrics for evaluating particle detection performance,
especially for cryoET particle picking.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from sklearn.metrics import precision_recall_curve, average_precision_score


def calculate_distances(detected: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """
    Calculate distances between detected particles and ground truth particles.
    
    Args:
        detected: array of detected particle coordinates (N, 3) or (N, 2)
        ground_truth: array of ground truth particle coordinates (M, 3) or (M, 2)
        
    Returns:
        distances: 2D array of distances between all detected and ground truth particles (N, M)
    """
    # Check inputs
    if detected.shape[0] == 0 or ground_truth.shape[0] == 0:
        return np.zeros((detected.shape[0], ground_truth.shape[0]))
    
    # Ensure both arrays have the same number of dimensions
    if detected.shape[1] != ground_truth.shape[1]:
        raise ValueError(f"Dimension mismatch: detected has shape {detected.shape}, ground_truth has shape {ground_truth.shape}")
    
    # Calculate distances between all pairs
    distances = np.zeros((detected.shape[0], ground_truth.shape[0]))
    for i, det in enumerate(detected):
        for j, gt in enumerate(ground_truth):
            distances[i, j] = np.sqrt(np.sum((det - gt) ** 2))
    
    return distances


def calculate_precision_recall_f1(
    detected: np.ndarray, 
    ground_truth: np.ndarray, 
    tolerance: float = 10.0
) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score for particle detection.
    
    Args:
        detected: array of detected particle coordinates (N, 3) or (N, 2)
        ground_truth: array of ground truth particle coordinates (M, 3) or (M, 2)
        tolerance: maximum distance (in pixels/voxels) for a detection to be considered correct
        
    Returns:
        tuple: (precision, recall, F1 score)
    """
    if detected.shape[0] == 0:
        return 0.0, 0.0, 0.0
    
    if ground_truth.shape[0] == 0:
        return 0.0, 0.0, 0.0
    
    # Calculate distances between all detections and ground truth
    distances = calculate_distances(detected, ground_truth)
    
    # Find matches using greedy assignment
    true_positives = 0
    matched_gt = set()
    
    # For each detection, find the closest unmatched ground truth within tolerance
    for i in range(distances.shape[0]):
        min_idx = np.argmin(distances[i])
        min_dist = distances[i, min_idx]
        
        if min_dist <= tolerance and min_idx not in matched_gt:
            true_positives += 1
            matched_gt.add(min_idx)
    
    # Calculate metrics
    precision = true_positives / detected.shape[0] if detected.shape[0] > 0 else 0.0
    recall = true_positives / ground_truth.shape[0] if ground_truth.shape[0] > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def calculate_average_precision(
    detected: np.ndarray, 
    scores: np.ndarray,
    ground_truth: np.ndarray, 
    tolerance: float = 10.0
) -> Tuple[float, List[float], List[float], List[float]]:
    """
    Calculate average precision (AP) for particle detection.
    
    Args:
        detected: array of detected particle coordinates (N, 3) or (N, 2)
        scores: confidence scores for each detection (N,)
        ground_truth: array of ground truth particle coordinates (M, 3) or (M, 2)
        tolerance: maximum distance (in pixels/voxels) for a detection to be considered correct
        
    Returns:
        tuple: (average precision, precision values, recall values, thresholds)
    """
    if detected.shape[0] == 0 or ground_truth.shape[0] == 0:
        return 0.0, [], [], []
    
    # Calculate distances between all detections and ground truth
    distances = calculate_distances(detected, ground_truth)
    
    # For each detection, check if it's a true positive at minimum distance
    y_true = np.zeros(detected.shape[0], dtype=bool)
    matched_gt = set()
    
    # Sort detections by score in descending order
    sort_indices = np.argsort(-scores)
    sorted_dists = distances[sort_indices]
    
    # Find matches for sorted detections
    for i in range(sorted_dists.shape[0]):
        min_idx = np.argmin(sorted_dists[i])
        min_dist = sorted_dists[i, min_idx]
        
        if min_dist <= tolerance and min_idx not in matched_gt:
            y_true[sort_indices[i]] = True
            matched_gt.add(min_idx)
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    
    # Calculate average precision
    ap = average_precision_score(y_true, scores)
    
    return ap, precision.tolist(), recall.tolist(), thresholds.tolist()


def calculate_detector_metrics(
    detected: np.ndarray, 
    ground_truth: np.ndarray, 
    scores: Optional[np.ndarray] = None, 
    tolerance: float = 10.0
) -> Dict[str, Union[float, List[float]]]:
    """
    Calculate a comprehensive set of metrics for particle detection.
    
    Args:
        detected: array of detected particle coordinates (N, 3) or (N, 2)
        ground_truth: array of ground truth particle coordinates (M, 3) or (M, 2)
        scores: confidence scores for each detection (N,) (optional)
        tolerance: maximum distance (in pixels/voxels) for a detection to be considered correct
        
    Returns:
        dict: dictionary of metrics including precision, recall, F1, and AP
    """
    metrics = {}
    
    # Basic metrics
    precision, recall, f1 = calculate_precision_recall_f1(detected, ground_truth, tolerance)
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1_score"] = f1
    
    # Absolute counts
    metrics["num_detections"] = detected.shape[0]
    metrics["num_ground_truth"] = ground_truth.shape[0]
    
    # Calculate true positives, false positives, and false negatives
    if detected.shape[0] > 0 and ground_truth.shape[0] > 0:
        distances = calculate_distances(detected, ground_truth)
        matched_gt = set()
        true_positives = 0
        
        for i in range(distances.shape[0]):
            min_idx = np.argmin(distances[i])
            min_dist = distances[i, min_idx]
            
            if min_dist <= tolerance and min_idx not in matched_gt:
                true_positives += 1
                matched_gt.add(min_idx)
        
        false_positives = detected.shape[0] - true_positives
        false_negatives = ground_truth.shape[0] - true_positives
        
        metrics["true_positives"] = true_positives
        metrics["false_positives"] = false_positives
        metrics["false_negatives"] = false_negatives
    
    # Calculate average precision if scores are provided
    if scores is not None and scores.shape[0] == detected.shape[0]:
        ap, precision_values, recall_values, thresholds = calculate_average_precision(
            detected, scores, ground_truth, tolerance
        )
        metrics["average_precision"] = ap
        metrics["precision_values"] = precision_values
        metrics["recall_values"] = recall_values
        metrics["thresholds"] = thresholds
    
    return metrics


def calculate_dog_detector_metrics(
    volume: np.ndarray,
    ground_truth: np.ndarray,
    detector,
    tolerance: float = 10.0
) -> Dict[str, Union[float, List[float]]]:
    """
    Calculate metrics for DoG detector by running the detector on the volume.
    
    Args:
        volume: 3D volume to detect particles in
        ground_truth: array of ground truth particle coordinates (M, 3)
        detector: DoG detector instance
        tolerance: maximum distance (in pixels/voxels) for a detection to be considered correct
        
    Returns:
        dict: dictionary of metrics including precision, recall, F1
    """
    # Run detector on volume
    detected, scores = detector.detect(volume, return_scores=True)
    
    # Calculate metrics
    metrics = calculate_detector_metrics(
        detected, ground_truth, scores=scores, tolerance=tolerance
    )
    
    return metrics
