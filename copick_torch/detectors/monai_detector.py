"""
MONAI-based particle detector for CryoET data.
This detector is based on MONAI's RetinaNet implementation.
"""

import os
import logging
import warnings
from typing import Dict, List, Tuple, Union, Optional, Any, Sequence

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from pathlib import Path

import monai
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.networks.retinanet_network import RetinaNet, resnet_fpn_feature_extractor
from monai.apps.detection.utils.anchor_utils import AnchorGenerator, AnchorGeneratorWithAnchorShape
from monai.apps.detection.utils.predict_utils import ensure_dict_value_to_list_
from monai.networks.nets import resnet
from monai.data.box_utils import box_iou
from monai.transforms import Compose, ScaleIntensity, EnsureChannelFirst, ToTensor, SpatialPad, Resize
from monai.inferers import SlidingWindowInferer
from monai.utils import ensure_tuple_rep, BlendMode, PytorchPadMode

class MONAIParticleDetector:
    """
    MONAI-based particle detector for cryoET data based on RetinaNet.
    
    This detector uses MONAI's RetinaNet implementation for 3D object detection
    and adapts it for particle picking in cryoET data.
    
    Args:
        spatial_dims: number of spatial dimensions (2 or 3)
        num_classes: number of output classes (particle types) to detect
        feature_size: size of features maps (32, 64, 128, etc.)
        anchor_sizes: list of anchor sizes in voxels (e.g., [(8,8,8), (16,16,16)])
        pretrained: whether to use pretrained backbone weights
        device: device to run the detector on ('cuda', 'cpu')
        sliding_window_size: size of sliding window for inference (must be divisible by 16)
        sliding_window_batch_size: batch size for sliding window inference
        sliding_window_overlap: overlap of sliding windows during inference
        detection_threshold: confidence threshold for detections
        nms_threshold: non-maximum suppression threshold for overlapping detections
        max_detections_per_volume: maximum number of detections per volume
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        num_classes: int = 1,
        feature_size: int = 32,
        anchor_sizes: Sequence[Sequence[int]] = None,
        pretrained: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sliding_window_size: Sequence[int] = None,
        sliding_window_batch_size: int = 4,
        sliding_window_overlap: float = 0.25,
        detection_threshold: float = 0.3,
        nms_threshold: float = 0.1,
        max_detections_per_volume: int = 2000,
    ):
        self.spatial_dims = spatial_dims
        self.num_classes = num_classes
        self.device = device
        
        # Set default anchor sizes if not provided
        if anchor_sizes is None:
            # Default sizes appropriate for cryoET particles
            if spatial_dims == 3:
                self.anchor_sizes = [(8, 8, 8), (16, 16, 16)]
            else:
                self.anchor_sizes = [(8, 8), (16, 16)]
        else:
            self.anchor_sizes = anchor_sizes

        # Set default sliding window size if not provided
        if sliding_window_size is None:
            # Default sliding window size (must be divisible by 16)
            if spatial_dims == 3:
                self.sliding_window_size = (64, 64, 64)
            else:
                self.sliding_window_size = (128, 128)
        else:
            self.sliding_window_size = sliding_window_size
            
        self.sliding_window_batch_size = sliding_window_batch_size
        self.sliding_window_overlap = sliding_window_overlap
        
        # Detection parameters
        self.detection_threshold = detection_threshold
        self.nms_threshold = nms_threshold
        self.max_detections_per_volume = max_detections_per_volume
        
        # Create the detector
        self.detector = self._create_detector(
            spatial_dims=spatial_dims,
            num_classes=num_classes,
            feature_size=feature_size,
            pretrained=pretrained
        )
        
        # Configure detector for inference
        self.detector.set_box_selector_parameters(
            score_thresh=self.detection_threshold,
            nms_thresh=self.nms_threshold,
            detections_per_img=self.max_detections_per_volume,
            apply_sigmoid=True
        )
        
        # Configure sliding window inference
        self.detector.set_sliding_window_inferer(
            roi_size=self.sliding_window_size,
            sw_batch_size=self.sliding_window_batch_size,
            overlap=self.sliding_window_overlap,
            mode=BlendMode.CONSTANT,
            padding_mode=PytorchPadMode.CONSTANT,
            cval=0.0,
            sw_device=self.device,
            device=self.device,
            progress=True
        )
        
        # Input pre-processing
        self.transform = Compose([
            EnsureChannelFirst(),
            ScaleIntensity(),
            ToTensor(),
        ])
        
        # Move model to device
        self.detector.to(self.device)
        self.detector.eval()  # Set to eval mode

    def _create_detector(self, spatial_dims, num_classes, feature_size, pretrained):
        """
        Create the MONAI RetinaNet detector.
        
        Args:
            spatial_dims: number of spatial dimensions
            num_classes: number of classes to predict
            feature_size: size of feature maps
            pretrained: whether to use pretrained weights
            
        Returns:
            RetinaNetDetector: detector model
        """
        # Create ResNet backbone
        if spatial_dims == 3:
            conv1_t_stride = (2, 2, 2)
            conv1_t_size = (7, 7, 7)
        else:
            conv1_t_stride = (2, 2)
            conv1_t_size = (7, 7)
        
        # Use ResNet18 as the backbone for faster inference
        backbone = resnet.ResNet(
            spatial_dims=spatial_dims,
            block=resnet.ResNetBlock,
            layers=[2, 2, 2, 2],  # ResNet18 architecture
            block_inplanes=resnet.get_inplanes(),
            n_input_channels=1,  # Single channel for tomogram data
            conv1_t_stride=conv1_t_stride,
            conv1_t_size=conv1_t_size,
            num_classes=num_classes
        )
        
        # Define which layers to return from the FPN
        returned_layers = [1, 2, 3]
        
        # Create feature extractor with FPN
        feature_extractor = resnet_fpn_feature_extractor(
            backbone=backbone,
            spatial_dims=spatial_dims,
            pretrained_backbone=pretrained,
            trainable_backbone_layers=None,
            returned_layers=returned_layers,
        )
        
        # Calculate required divisibility for network input
        size_divisible = tuple(2 * s * 2**max(returned_layers) for s in conv1_t_stride)
        
        # Create anchor generator with custom anchor sizes for cryoET particles
        anchor_generator = AnchorGeneratorWithAnchorShape(
            feature_map_scales=(1, 2, 4),  # Scales for different feature map levels
            base_anchor_shapes=self.anchor_sizes  # Use custom anchor sizes
        )
        
        # Create RetinaNet network
        network = RetinaNet(
            spatial_dims=spatial_dims,
            num_classes=num_classes,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
            feature_extractor=feature_extractor,
            size_divisible=size_divisible,
        )
        
        # Create RetinaNet detector
        detector = RetinaNetDetector(network, anchor_generator)
        
        return detector
    
    def load_weights(self, weights_path):
        """
        Load trained weights for the detector.
        
        Args:
            weights_path: path to the weights file
        """
        if os.path.exists(weights_path):
            print(f"Loading weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location=self.device)
            self.detector.network.load_state_dict(state_dict)
        else:
            warnings.warn(f"Weights file {weights_path} not found.")
    
    def save_weights(self, weights_path):
        """
        Save the detector weights.
        
        Args:
            weights_path: path to save the weights file
        """
        Path(weights_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.detector.network.state_dict(), weights_path)
        print(f"Saved weights to {weights_path}")
    
    def detect(self, volume, return_scores=False, use_inferer=True):
        """
        Detect particles in a 3D volume.
        
        Args:
            volume: 3D numpy array or torch tensor
            return_scores: whether to return detection scores
            use_inferer: whether to use sliding window inference 
            
        Returns:
            numpy array of particle coordinates [N, spatial_dims]
            (optional) scores for each detection [N]
        """
        # Ensure detector is in eval mode
        self.detector.eval()
        
        # Ensure input is preprocessed correctly
        if isinstance(volume, np.ndarray):
            # Apply transforms to numpy array
            input_tensor = self.transform(volume)
        elif isinstance(volume, torch.Tensor):
            # Ensure tensor is in correct format
            if volume.ndim == self.spatial_dims:
                # Add channel dimension if missing
                input_tensor = volume.unsqueeze(0)
            else:
                input_tensor = volume
                
            # Move to device
            input_tensor = input_tensor.to(self.device)
        else:
            raise ValueError(f"Unsupported input type: {type(volume)}")
        
        # Ensure input has batch dimension
        if input_tensor.dim() == self.spatial_dims + 1:  # [C, D, H, W] or [C, H, W]
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension [B, C, D, H, W] or [B, C, H, W]
            
        with torch.no_grad():
            # Run inference
            detections = self.detector.forward(input_tensor, use_inferer=use_inferer)
            
        # Extract coordinates
        coordinates_list = []
        scores_list = []
        
        for det in detections:
            boxes = det["boxes"].cpu().numpy()
            scores = det["labels_scores"].cpu().numpy()
            
            # Convert boxes [xmin, ymin, zmin, xmax, ymax, zmax] to coordinates [x, y, z]
            if boxes.shape[0] > 0:
                if self.spatial_dims == 3:
                    coordinates = np.zeros((boxes.shape[0], 3))
                    coordinates[:, 0] = (boxes[:, 0] + boxes[:, 3]) / 2  # x center
                    coordinates[:, 1] = (boxes[:, 1] + boxes[:, 4]) / 2  # y center
                    coordinates[:, 2] = (boxes[:, 2] + boxes[:, 5]) / 2  # z center
                else:
                    coordinates = np.zeros((boxes.shape[0], 2))
                    coordinates[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # x center
                    coordinates[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # y center
                
                coordinates_list.append(coordinates)
                scores_list.append(scores)
        
        # Concatenate results from all batches
        if coordinates_list:
            coordinates = np.concatenate(coordinates_list, axis=0)
            scores = np.concatenate(scores_list, axis=0)
        else:
            # No detections
            coordinates = np.zeros((0, self.spatial_dims))
            scores = np.zeros(0)
        
        if return_scores:
            return coordinates, scores
        return coordinates
    
    def train(self, 
              train_dataloader,
              val_dataloader=None,
              num_epochs=10,
              learning_rate=1e-4,
              weight_decay=1e-5,
              save_path=None,
              best_metric_name="val_loss"):
        """
        Train the detector.
        
        Args:
            train_dataloader: dataloader for training data
            val_dataloader: dataloader for validation data
            num_epochs: number of epochs to train
            learning_rate: learning rate for optimizer
            weight_decay: weight decay for optimizer
            save_path: path to save the best model weights
            best_metric_name: metric name to determine the best model
            
        Returns:
            dict: training metrics
        """
        # Set detector to training mode
        self.detector.train()
        
        # Define optimizer
        optimizer = torch.optim.AdamW(
            self.detector.network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Define learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, verbose=True
        )
        
        # Training metrics
        best_metric = float("inf") if "loss" in best_metric_name else 0.0
        metrics = {"train_loss": [], "val_loss": []}
        
        for epoch in range(num_epochs):
            # Training loop
            self.detector.train()
            epoch_loss = 0
            
            for batch_idx, batch_data in enumerate(train_dataloader):
                # Extract data and target
                images, targets = batch_data
                images = images.to(self.device)
                
                # Prepare targets for RetinaNet format
                formatted_targets = []
                for idx, target in enumerate(targets):
                    boxes = target.get("boxes", None)
                    labels = target.get("labels", None)
                    
                    if boxes is None or labels is None:
                        continue
                        
                    formatted_targets.append({
                        "boxes": boxes.to(self.device),
                        "labels": labels.to(self.device)
                    })
                
                # Skip batch if no valid targets
                if not formatted_targets:
                    continue
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                losses = self.detector(images, formatted_targets)
                loss = losses["classification"] + losses["box_regression"]
                
                # Backward pass
                loss.backward()
                
                # Update weights
                optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                
                # Print progress
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
            
            # Compute average loss for the epoch
            avg_train_loss = epoch_loss / len(train_dataloader)
            metrics["train_loss"].append(avg_train_loss)
            
            # Validation loop
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                metrics["val_loss"].append(val_loss)
                
                # Update learning rate
                scheduler.step(val_loss)
                
                # Save best model
                if save_path is not None:
                    if ("loss" in best_metric_name and val_loss < best_metric) or \
                       ("loss" not in best_metric_name and val_loss > best_metric):
                        best_metric = val_loss
                        self.save_weights(save_path)
                        print(f"Saved best model with {best_metric_name} = {best_metric:.4f}")
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}" + 
                  (f", Val Loss: {val_loss:.4f}" if val_dataloader else ""))
        
        # Set back to eval mode
        self.detector.eval()
        
        return metrics
    
    def validate(self, val_dataloader):
        """
        Validate the detector on validation data.
        
        Args:
            val_dataloader: dataloader for validation data
            
        Returns:
            float: validation loss
        """
        # Set detector to evaluation mode
        self.detector.eval()
        
        # Validation metrics
        val_loss = 0
        
        with torch.no_grad():
            for batch_data in val_dataloader:
                # Extract data and target
                images, targets = batch_data
                images = images.to(self.device)
                
                # Prepare targets for RetinaNet format
                formatted_targets = []
                for idx, target in enumerate(targets):
                    boxes = target.get("boxes", None)
                    labels = target.get("labels", None)
                    
                    if boxes is None or labels is None:
                        continue
                        
                    formatted_targets.append({
                        "boxes": boxes.to(self.device),
                        "labels": labels.to(self.device)
                    })
                
                # Skip batch if no valid targets
                if not formatted_targets:
                    continue
                
                # Forward pass
                losses = self.detector(images, formatted_targets)
                loss = losses["classification"] + losses["box_regression"]
                
                # Update metrics
                val_loss += loss.item()
        
        # Compute average loss for the validation set
        avg_val_loss = val_loss / len(val_dataloader)
        
        return avg_val_loss
