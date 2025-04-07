"""
Example script demonstrating the MONAI-based particle detector.

This script shows how to use the MONAI-based detector for particle picking in cryoET tomograms.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import logging
import argparse

from copick_torch.detectors.monai_detector import MONAIParticleDetector
from copick_torch.metrics import calculate_detector_metrics
from copick_torch.dataloaders import CryoETDataPortalDataset
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

def train_on_dataset(dataset_ids, overlay_root, output_dir, num_epochs=10, batch_size=4):
    """
    Train a particle detector on the specified datasets.
    
    Args:
        dataset_ids: list of dataset IDs from the CryoET Data Portal
        overlay_root: root URL for the overlay storage
        output_dir: directory to save model weights and outputs
        num_epochs: number of epochs to train for
        batch_size: batch size for training
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset
    logger.info(f"Creating dataset for IDs: {dataset_ids}")
    dataset = CryoETParticleDataset(
        dataset_ids=dataset_ids,
        overlay_root=overlay_root,
        boxsize=(64, 64, 64),
        voxel_spacing=10.0,
        include_background=True,
        background_ratio=0.5,
        min_background_distance=20.0,
        cache_dir=os.path.join(output_dir, 'cache')
    )
    
    # Initialize dataset (this will load data)
    logger.info("Initializing dataset...")
    dataset.initialize()
    
    # Create dataloader
    logger.info("Creating dataloader...")
    dataloader = dataset.get_dataloader(batch_size=batch_size, shuffle=True)
    
    # Create detector
    logger.info("Creating MONAI detector...")
    detector = MONAIParticleDetector(
        spatial_dims=3,
        num_classes=1,
        feature_size=32,
        anchor_sizes=[(8, 8, 8), (16, 16, 16), (32, 32, 32)],
        device="cuda" if torch.cuda.is_available() else "cpu",
        sliding_window_size=(64, 64, 64),
        sliding_window_batch_size=batch_size,
        sliding_window_overlap=0.25,
        detection_threshold=0.3,
        nms_threshold=0.1,
        max_detections_per_volume=1000
    )
    
    # Train the detector
    logger.info("Training detector...")
    detector.train(
        train_dataloader=dataloader,
        num_epochs=num_epochs,
        learning_rate=1e-4,
        weight_decay=1e-5,
        save_path=os.path.join(output_dir, 'model_weights.pt')
    )
    
    logger.info(f"Training complete. Model saved to {os.path.join(output_dir, 'model_weights.pt')}")

def inference_on_dataset(dataset_ids, overlay_root, weights_path, output_dir):
    """
    Run inference on the specified datasets.
    
    Args:
        dataset_ids: list of dataset IDs from the CryoET Data Portal
        overlay_root: root URL for the overlay storage
        weights_path: path to trained model weights
        output_dir: directory to save outputs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset
    logger.info(f"Creating dataset for IDs: {dataset_ids}")
    dataset = CryoETDataPortalDataset(
        dataset_ids=dataset_ids,
        overlay_root=overlay_root,
        voxel_spacing=10.0,
        cache_dir=os.path.join(output_dir, 'cache')
    )
    
    # Create detector
    logger.info("Creating MONAI detector...")
    detector = MONAIParticleDetector(
        spatial_dims=3,
        num_classes=1,
        feature_size=32,
        anchor_sizes=[(8, 8, 8), (16, 16, 16), (32, 32, 32)],
        device="cuda" if torch.cuda.is_available() else "cpu",
        sliding_window_size=(64, 64, 64),
        sliding_window_batch_size=4,
        sliding_window_overlap=0.25,
        detection_threshold=0.3,
        nms_threshold=0.1,
        max_detections_per_volume=1000
    )
    
    # Load weights
    if os.path.exists(weights_path):
        logger.info(f"Loading weights from {weights_path}")
        detector.load_weights(weights_path)
    else:
        logger.warning(f"Weights file {weights_path} not found, using untrained model")
    
    # Process each tomogram in the dataset
    all_metrics = []
    
    for idx in range(len(dataset)):
        # Load tomogram
        logger.info(f"Processing tomogram {idx+1}/{len(dataset)}")
        tomogram, metadata = dataset[idx]
        
        # Get ground truth picks
        ground_truth = dataset.get_picks_for_tomogram(idx)
        
        # Run inference
        logger.info("Running inference...")
        if isinstance(tomogram, torch.Tensor):
            tomogram_np = tomogram.numpy()
            if tomogram_np.ndim == 4:  # Remove channel dimension if present
                tomogram_np = tomogram_np[0]
        else:
            tomogram_np = tomogram
            
        picks, scores = detector.detect(tomogram_np, return_scores=True, use_inferer=True)
        logger.info(f"Detected {len(picks)} particles")
        
        # Calculate metrics if ground truth is available
        if ground_truth is not None and len(ground_truth) > 0:
            logger.info(f"Calculating metrics against {len(ground_truth)} ground truth particles")
            metrics = calculate_detector_metrics(
                picks, ground_truth, scores, tolerance=10.0
            )
            
            logger.info(f"Precision: {metrics['precision']:.3f}")
            logger.info(f"Recall: {metrics['recall']:.3f}")
            logger.info(f"F1 Score: {metrics['f1_score']:.3f}")
            logger.info(f"Average Precision: {metrics.get('average_precision', 'N/A')}")
            
            all_metrics.append(metrics)
        
        # Plot and save results
        logger.info("Plotting results...")
        plot_results(tomogram_np, picks, ground_truth)
        plt.savefig(os.path.join(output_dir, f'tomo_{idx}_results.png'))
    
    # Save overall metrics if available
    if all_metrics:
        # Calculate average metrics
        avg_metrics = {
            'precision': np.mean([m['precision'] for m in all_metrics]),
            'recall': np.mean([m['recall'] for m in all_metrics]),
            'f1_score': np.mean([m['f1_score'] for m in all_metrics]),
        }
        
        if 'average_precision' in all_metrics[0]:
            avg_metrics['average_precision'] = np.mean([m['average_precision'] for m in all_metrics])
        
        # Save to file
        with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
            f.write(f"Average Metrics across {len(all_metrics)} tomograms:\n")
            f.write(f"Precision: {avg_metrics['precision']:.3f}\n")
            f.write(f"Recall: {avg_metrics['recall']:.3f}\n")
            f.write(f"F1 Score: {avg_metrics['f1_score']:.3f}\n")
            if 'average_precision' in avg_metrics:
                f.write(f"Average Precision: {avg_metrics['average_precision']:.3f}\n")

def inference_on_file(config_path, weights_path, output_dir):
    """
    Run inference on a local copick configuration.
    
    Args:
        config_path: path to the copick configuration file
        weights_path: path to trained model weights
        output_dir: directory to save outputs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load copick project
    logger.info(f"Loading copick project from {config_path}")
    root = copick.from_file(config_path)
    
    # Create detector
    logger.info("Creating MONAI detector...")
    detector = MONAIParticleDetector(
        spatial_dims=3,
        num_classes=1,
        feature_size=32,
        anchor_sizes=[(8, 8, 8), (16, 16, 16), (32, 32, 32)],
        device="cuda" if torch.cuda.is_available() else "cpu",
        sliding_window_size=(64, 64, 64),
        sliding_window_batch_size=4,
        sliding_window_overlap=0.25,
        detection_threshold=0.3,
        nms_threshold=0.1,
        max_detections_per_volume=1000
    )
    
    # Load weights
    if os.path.exists(weights_path):
        logger.info(f"Loading weights from {weights_path}")
        detector.load_weights(weights_path)
    else:
        logger.warning(f"Weights file {weights_path} not found, using untrained model")
    
    # Process each run in the project
    for run_idx, run in enumerate(root.runs):
        logger.info(f"Processing run {run_idx+1}/{len(root.runs)}: {run.name}")
        
        # Get all voxel spacings
        for vs_idx, vs in enumerate(run.voxel_spacings):
            logger.info(f"Processing voxel spacing {vs_idx+1}/{len(run.voxel_spacings)}: {vs.voxel_size}")
            
            # Get all tomograms
            for tomo_idx, tomo in enumerate(vs.tomograms):
                logger.info(f"Processing tomogram {tomo_idx+1}/{len(vs.tomograms)}: {tomo.tomo_type}")
                
                # Load the tomogram
                try:
                    logger.info("Loading tomogram data...")
                    tomogram_array = tomo.numpy()
                except Exception as e:
                    logger.error(f"Error loading tomogram: {e}")
                    continue
                
                # Run inference
                logger.info("Running inference...")
                picks, scores = detector.detect(tomogram_array, return_scores=True, use_inferer=True)
                logger.info(f"Detected {len(picks)} particles")
                
                # Get ground truth if available
                ground_truth = None
                picks_sets = run.get_picks()
                
                if picks_sets:
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
                        
                        # Save metrics
                        metrics_path = os.path.join(output_dir, f'run{run_idx}_vs{vs_idx}_tomo{tomo_idx}_metrics.txt')
                        with open(metrics_path, 'w') as f:
                            for key, value in metrics.items():
                                if not isinstance(value, list):
                                    f.write(f"{key}: {value}\n")
                
                # Plot and save results
                logger.info("Plotting results...")
                plot_results(tomogram_array, picks, ground_truth)
                plt.savefig(os.path.join(output_dir, f'run{run_idx}_vs{vs_idx}_tomo{tomo_idx}_results.png'))
                
                # Save the picks
                picks_path = os.path.join(output_dir, f'run{run_idx}_vs{vs_idx}_tomo{tomo_idx}_picks.npy')
                np.save(picks_path, picks)
                logger.info(f"Saved picks to {picks_path}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="MONAI Particle Detector Example")
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Mode to run')
    
    # Train mode
    train_parser = subparsers.add_parser('train', help='Train a model on a dataset')
    train_parser.add_argument('--dataset-ids', type=int, nargs='+', required=True, 
                            help='Dataset IDs from the CryoET Data Portal')
    train_parser.add_argument('--overlay-root', type=str, required=True,
                            help='Root URL for the overlay storage')
    train_parser.add_argument('--output-dir', type=str, default='./output',
                            help='Directory to save outputs')
    train_parser.add_argument('--epochs', type=int, default=10,
                            help='Number of epochs to train for')
    train_parser.add_argument('--batch-size', type=int, default=4,
                            help='Batch size for training')
    
    # Inference on dataset mode
    inference_dataset_parser = subparsers.add_parser('inference-dataset', 
                                                   help='Run inference on a dataset')
    inference_dataset_parser.add_argument('--dataset-ids', type=int, nargs='+', required=True, 
                                        help='Dataset IDs from the CryoET Data Portal')
    inference_dataset_parser.add_argument('--overlay-root', type=str, required=True,
                                        help='Root URL for the overlay storage')
    inference_dataset_parser.add_argument('--weights', type=str, required=True,
                                        help='Path to trained model weights')
    inference_dataset_parser.add_argument('--output-dir', type=str, default='./output',
                                        help='Directory to save outputs')
    
    # Inference on file mode
    inference_file_parser = subparsers.add_parser('inference-file', 
                                                help='Run inference on a local copick configuration')
    inference_file_parser.add_argument('--config', type=str, required=True,
                                     help='Path to the copick configuration file')
    inference_file_parser.add_argument('--weights', type=str, required=True,
                                     help='Path to trained model weights')
    inference_file_parser.add_argument('--output-dir', type=str, default='./output',
                                     help='Directory to save outputs')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the specified mode
    if args.mode == 'train':
        train_on_dataset(
            dataset_ids=args.dataset_ids,
            overlay_root=args.overlay_root,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size
        )
    elif args.mode == 'inference-dataset':
        inference_on_dataset(
            dataset_ids=args.dataset_ids,
            overlay_root=args.overlay_root,
            weights_path=args.weights,
            output_dir=args.output_dir
        )
    elif args.mode == 'inference-file':
        inference_on_file(
            config_path=args.config,
            weights_path=args.weights,
            output_dir=args.output_dir
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
