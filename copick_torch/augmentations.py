import random
import torch
import numpy as np

class MixupAugmentation:
    """
    Implements Mixup augmentation for 3D volumes.
    
    Mixup is a data augmentation technique that creates virtual training examples
    by mixing pairs of inputs and their labels with a random proportion.
    
    Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
    https://arxiv.org/abs/1710.09412
    """
    
    def __init__(self, alpha=0.2):
        """
        Initialize the Mixup augmentation.
        
        Args:
            alpha: Parameter for Beta distribution. Higher values result in more mixing.
        """
        self.alpha = alpha
        
    def __call__(self, images, labels):
        """
        Apply mixup augmentation to a batch of images and labels.
        
        Args:
            images: Tensor of shape [batch_size, channels, depth, height, width]
            labels: Tensor of shape [batch_size]
            
        Returns:
            Tuple of (mixed_images, label_a, label_b, lam) where:
                - mixed_images: The mixup result
                - label_a: Original labels
                - label_b: Mixed-in labels
                - lam: Mixing coefficient from Beta distribution
        """
        batch_size = images.size(0)
        
        # Sample mixing coefficient from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
            
        # Ensure lam is within reasonable bounds
        lam = max(lam, 1 - lam)
        
        # Generate random indices for mixing
        index = torch.randperm(batch_size, device=images.device)
        
        # Mix the images
        mixed_images = lam * images + (1 - lam) * images[index]
        
        # Return the mixed images and label information
        return mixed_images, labels, labels[index], lam
    
    @staticmethod
    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        """
        Apply mixup to the loss calculation.
        
        Args:
            criterion: Loss function
            pred: Model predictions
            y_a: First labels
            y_b: Second (mixed-in) labels
            lam: Mixing coefficient
            
        Returns:
            Mixed loss
        """
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
