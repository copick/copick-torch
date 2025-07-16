from collections import Counter

import numpy as np
import torch
from torch.utils.data import Sampler


class ClassBalancedSampler(Sampler):
    """
    A sampler that balances class distributions during training.

    This sampler is designed to address the class imbalance problem by providing
    a way to balance the frequency of each class in the mini-batches.
    """

    def __init__(self, labels, num_samples=None, replacement=True):
        """
        Initialize the class-balanced sampler.

        Args:
            labels: List or tensor of integer class labels for each sample
            num_samples: Number of samples to draw (default: len(labels))
            replacement: Whether to sample with replacement (default: True)
        """
        self.labels = np.array(labels)
        self.num_samples = len(labels) if num_samples is None else num_samples
        self.replacement = replacement

        # Count occurrences of each class
        label_counter = Counter(self.labels)
        self.class_counts = label_counter

        # Calculate sampling weights
        weight_per_class = {class_idx: 1.0 / count for class_idx, count in label_counter.items()}
        self.weights = np.array([weight_per_class[label] for label in self.labels])

        # Normalize weights to sum to 1
        self.weights = self.weights / self.weights.sum()

    def __iter__(self):
        """
        Generate a random sequence of indices based on weighted sampling.

        Returns:
            Iterator over indices
        """
        # Generate random indices weighted by class distribution
        indices = np.random.choice(len(self.labels), size=self.num_samples, replace=self.replacement, p=self.weights)

        return iter(indices.tolist())

    def __len__(self):
        """
        Return the number of samples in the sampler.

        Returns:
            Number of samples
        """
        return self.num_samples
