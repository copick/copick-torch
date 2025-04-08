# copick-torch

[![codecov](https://codecov.io/gh/copick/copick-torch/branch/main/graph/badge.svg)](https://codecov.io/gh/copick/copick-torch)

Torch utilities for [copick](https://github.com/copick/copick)

## Quick demo

```bash
# Simple training example
uv run examples/simple_training.py

# Fourier augmentation demo
uv run examples/fourier_augmentation_demo.py

# SplicedMixup with Gaussian blur visualization
uv run examples/spliced_mixup_example.py

# SplicedMixup with Fourier augmentation visualization
uv run examples/spliced_mixup_fourier_example.py
```

## Installation

```bash
pip install copick-torch
```

## Features

### Data augmentation with MONAI

copick-torch now leverages the MONAI framework for robust medical image augmentations. It provides:

1. A wide range of intensity-based augmentations:
   - GaussianNoise, RicianNoise
   - GibbsNoise, KSpaceSpikeNoise
   - Scaling, contrast adjustment
   - Histogram shifting
   - Gaussian smoothing/sharpening

2. Specialized augmentations for cryo-ET data:
   - FourierAugment3D: Frequency domain augmentations
   - MixupAugmentation: For synthesizing new training examples

3. Flexible augmentation pipeline:
   - AugmentationFactory for easily creating transform chains
   - Fallback implementations when MONAI isn't available

### Dataset classes

1. **SimpleCopickDataset**: Efficient dataset for training with copick data
   - Automatic data loading and caching
   - Background sampling
   - Configurable augmentations

2. **SplicedMixupDataset**: Advanced dataset for synthetic data integration
   - Combines experimental and synthetic data
   - In-memory zarr array handling
   - Gaussian boundary blending

## Usage

### Basic augmentation example

```python
from copick_torch.augmentations import AugmentationFactory

# Create a transform pipeline with specified augmentations
transforms = AugmentationFactory.create_transforms(
    augmentation_types=["gaussian_noise", "scale_intensity", "fourier"],
    prob=0.5  # 50% chance of applying each transform
)

# Apply to your data
augmented_volume = transforms(volume)
```

### Advanced usage with multiple augmentations

```python
import numpy as np
from copick_torch.augmentations import AugmentationFactory, FourierAugment3D, MixupAugmentation

# Create a synthetic test volume
volume = np.random.randn(64, 64, 64)

# Create various augmentation transforms
transforms = AugmentationFactory.create_transforms([
    "gaussian_noise",  
    "rician_noise",
    "gibbs_noise",
    "gaussian_smooth",
    "gaussian_sharpen"
], prob=0.3)

# Apply transforms
augmented_volume = transforms(volume)

# For training with mixup
mixup = MixupAugmentation(alpha=0.2)
mixed_images, label_a, label_b, lam = mixup(images_batch, labels_batch)
```

See the examples directory for more usage examples.
