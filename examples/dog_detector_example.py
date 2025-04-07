"""
Example script demonstrating the Difference of Gaussian (DoG) particle detector.

This script shows how to use the DoG detector for particle picking in cryoET tomograms.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from copick_torch.detectors.dog_detector import DoGParticleDetector
from copick_torch.metrics import calculate_detector_metrics
import copick

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_results(tomogram, picks, ground_truth=None, slice_idx=None):
    """
    Plot detection results on a tomogram slice.
    
    Args:
        tomogram: 3D numpy array of the tomogram
        picks: detected particle coordinates (N, 3)
        ground_truth: optional ground truth particle coordinates (M, 3)
        slice_idx: optional slice index to plot, if None will use middle slice
    """
    if slice_idx is None:
        # Use middle slice
        slice_idx = tomogram.shape[0] // 2
    
    # Plot the slice
    plt.figure(figsize=(10, 8))
    plt.imshow(tomogram[slice_idx], cmap='gray')
    
    # Find particles in this slice (within +/- 3 slices)
    z_min, z_max = slice_idx - 3, slice_idx + 3
    
    # Plot detected particles
    if picks is not None and len(picks) > 0:
        picks_in_slice = picks[(picks[:, 0] >= z_min) & (picks[:, 0] <= z_max)]
        if len(picks_in_slice) > 0:
            plt.scatter(picks_in_slice[:, 2], picks_in_slice[:, 1], 
                        s=30, c='red', marker='o', alpha=0.7, label=f'Detected ({len(picks_in_slice)})')
    
    # Plot ground truth if provided
    if ground_truth is not None and len(ground_truth) > 0:
        gt_in_slice = ground_truth[(ground_truth[:, 0] >= z_min) & (ground_truth[:, 0] <= z_max)]
        if len(gt_in_slice) > 0:
            plt.scatter(gt_in_slice[:, 2], gt_in_slice[:, 1], 
                        s=30, c='blue', marker='x', alpha=0.7, label=f'Ground Truth ({len(gt_in_slice)})')
    
    plt.title(f'Tomogram Slice {slice_idx}')
    plt.colorbar(label='Intensity')
    if (picks is not None and len(picks) > 0) or (ground_truth is not None and len(ground_truth) > 0):
        plt.legend()
    plt.tight_layout()

def main():
    """Main function demonstrating the DoG detector."""
    # Check if cache directory exists, if not create it
    cache_dir = Path('cache')
    cache_dir.mkdir(exist_ok=True)
    
    # Define path to the copick project configuration
    config_path = './examples/czii_object_detection_training.json'
    
    # Check if the config exists, if not, print instructions
    if not os.path.exists(config_path):
        logger.error(f"Config file {config_path} not found!")
        logger.info("Please create a copick configuration file or specify an existing one.")
        return
    
    logger.info(f"Loading copick project from {config_path}")
    root = copick.from_file(config_path)
    
    # Get the first run
    if not root.runs:
        logger.error("No runs found in the copick project!")
        return
    
    run = root.runs[0]
    logger.info(f"Using run: {run.name}")
    
    # Get the first voxel spacing
    if not run.voxel_spacings:
        logger.error("No voxel spacings found in the run!")
        return
    
    voxel_spacing = run.voxel_spacings[0]
    logger.info(f"Using voxel spacing: {voxel_spacing.voxel_size}")
    
    # Get the first tomogram
    if not voxel_spacing.tomograms:
        logger.error("No tomograms found for this voxel spacing!")
        return
    
    tomogram = voxel_spacing.tomograms[0]
    logger.info(f"Using tomogram: {tomogram.tomo_type}")
    
    # Load the tomogram
    logger.info("Loading tomogram data...")
    tomogram_array = tomogram.numpy()
    
    # Create the DoG detector
    logger.info("Creating DoG detector...")
    detector = DoGParticleDetector(
        sigma1=1.0,
        sigma2=3.0,
        threshold_abs=0.1,
        min_distance=5,
        normalize=True,
        prefilter="median",
        prefilter_size=1.0
    )
    
    # Detect particles
    logger.info("Detecting particles...")
    picks, scores = detector.detect(tomogram_array, return_scores=True)
    logger.info(f"Detected {len(picks)} particles")
    
    # Get ground truth picks if available
    picks_sets = run.get_picks()
    ground_truth = None
    
    if picks_sets:
        logger.info(f"Found {len(picks_sets)} pick sets")
        # Combine all pick sets
        all_picks = []
        for pick_set in picks_sets:
            try:
                points, _ = pick_set.numpy()
                all_picks.append(points)
            except Exception as e:
                logger.error(f"Error loading picks: {e}")
        
        if all_picks:
            ground_truth = np.vstack(all_picks)
            logger.info(f"Loaded {len(ground_truth)} ground truth picks")
            
            # Calculate metrics
            logger.info("Calculating metrics...")
            metrics = calculate_detector_metrics(
                picks, ground_truth, scores, tolerance=10.0
            )
            
            logger.info(f"Precision: {metrics['precision']:.3f}")
            logger.info(f"Recall: {metrics['recall']:.3f}")
            logger.info(f"F1 Score: {metrics['f1_score']:.3f}")
            logger.info(f"Average Precision: {metrics.get('average_precision', 'N/A')}")
    
    # Plot results
    logger.info("Plotting results...")
    plot_results(tomogram_array, picks, ground_truth)
    
    # Save figure
    plt.savefig('dog_detector_results.png')
    logger.info("Saved figure to dog_detector_results.png")
    
    # Optimize detector parameters if ground truth is available
    if ground_truth is not None and len(ground_truth) > 0:
        logger.info("Optimizing detector parameters...")
        best_params = detector.optimize_parameters(
            tomogram_array,
            ground_truth,
            sigma1_range=(0.5, 2.0, 0.5),
            sigma2_range=(1.5, 4.0, 0.5),
            threshold_range=(0.05, 0.2, 0.05),
            min_distance_range=(3, 7, 2),
            tolerance=10.0
        )
        
        logger.info("Best parameters:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")
        
        # Update detector with best parameters
        detector.sigma1 = best_params['sigma1']
        detector.sigma2 = best_params['sigma2']
        detector.threshold_abs = best_params['threshold_abs']
        detector.min_distance = best_params['min_distance']
        
        # Detect particles with optimized parameters
        logger.info("Detecting particles with optimized parameters...")
        picks_opt, scores_opt = detector.detect(tomogram_array, return_scores=True)
        logger.info(f"Detected {len(picks_opt)} particles")
        
        # Plot results with optimized parameters
        logger.info("Plotting results with optimized parameters...")
        plot_results(tomogram_array, picks_opt, ground_truth)
        
        # Save figure
        plt.savefig('dog_detector_optimized_results.png')
        logger.info("Saved figure to dog_detector_optimized_results.png")

if __name__ == "__main__":
    main()
