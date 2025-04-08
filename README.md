# copick-torch

[![codecov](https://codecov.io/gh/copick/copick-torch/branch/main/graph/badge.svg)](https://codecov.io/gh/copick/copick-torch)

Torch utilities for [copick](https://github.com/copick/copick)

## Quick demo

```bash
# Simple training example
uv run examples/simple_training.py

# Fourier augmentation demo
uv run examples/fourier_augmentation_demo.py

# MONAI-based augmentation demo
uv run examples/monai_augmentation_demo.py

# SplicedMixup with Gaussian blur visualization
uv run examples/spliced_mixup_example.py

# SplicedMixup with Fourier augmentation visualization
uv run examples/spliced_mixup_fourier_example.py

# Generate augmentation documentation
python scripts/generate_augmentation_docs.py
```

## Features

### Augmentations

`copick-torch` includes various MONAI-based data augmentation techniques for 3D tomographic data:

- **MixupTransform**: MONAI-compatible implementation of the Mixup technique (Zhang et al., 2018), creating virtual training examples by mixing pairs of inputs and their labels with a random proportion.
- **FourierAugment3D**: MONAI-compatible implementation of Fourier-based augmentation that operates in the frequency domain, including random frequency dropout, phase noise injection, and intensity scaling.
- **AugmentationComposer**: A convenient wrapper for MONAI's augmentation transforms that makes it easy to build complex augmentation pipelines with intensity and spatial transforms.

Example usage of MONAI-based augmentations:

```python
# Import augmentations
from copick_torch.augmentations import FourierAugment3D, AugmentationComposer, MixupTransform

# Create a Fourier augmenter
fourier_aug = FourierAugment3D(
    freq_mask_prob=0.3,        # Probability of masking frequency components
    phase_noise_std=0.1,       # Standard deviation of phase noise
    intensity_scaling_range=(0.8, 1.2),  # Range for random intensity scaling
    prob=1.0                   # Probability of applying the transform
)

# Create an augmentation composer for multiple transforms
augmentation_composer = AugmentationComposer(
    intensity_transforms=True,  # Include intensity transforms
    spatial_transforms=True,    # Include spatial transforms
    prob_intensity=0.7,         # Probability of applying each intensity transform
    prob_spatial=0.5,           # Probability of applying each spatial transform
    rotate_range=0.1,           # Range for random rotation (radians)
    scale_range=0.15,           # Range for random scaling
    noise_std=0.1,              # Standard deviation for Gaussian noise
    gamma_range=(0.7, 1.3),     # Range for contrast adjustment
    intensity_range=(0.8, 1.2), # Range for intensity scaling
    shift_range=(-0.1, 0.1)     # Range for intensity shifting
)

# Create a mixup augmenter for training
mixup = MixupTransform(
    alpha=0.2,                 # Parameter for Beta distribution
    prob=1.0                   # Probability of applying mixup
)

# Apply augmentations to your data
augmented_volume_fourier = fourier_aug(volume_tensor)
augmented_volume_composer = augmentation_composer(volume_tensor)

# For mixup during training
mixed_images, label_a, label_b, lam = mixup(batch_images, batch_labels)
```

The augmentations are built on top of MONAI's transform framework, making them compatible with MONAI's other transforms and pipelines.

### Documentation

See the [docs directory](./docs) for documentation and examples:

- [Augmentation Examples](./docs/augmentation_examples): Visualizations of various augmentations applied to different classes from the dataset used in the `spliced_mixup_example.py` example.

## Citation

If you use `copick-torch` in your research, please cite:

```bibtex
@article{harrington2024open,
  title={Open-source Tools for CryoET Particle Picking Machine Learning Competitions},
  author={Harrington, Kyle I. and Zhao, Zhuowen and Schwartz, Jonathan and Kandel, Saugat and Ermel, Utz and Paraan, Mohammadreza and Potter, Clinton and Carragher, Bridget},
  journal={bioRxiv},
  year={2024},
  doi={10.1101/2024.11.04.621608}
}
```

This software was introduced in a NeurIPS 2024 Workshop on Machine Learning in Structural Biology as "Open-source Tools for CryoET Particle Picking Machine Learning Competitions".

## Development

### Install development dependencies

```bash
pip install ".[test]"
```

### Run tests

```bash
pytest
```

### View coverage report

```bash
# Generate terminal, HTML and XML coverage reports
pytest --cov=copick_torch --cov-report=term --cov-report=html --cov-report=xml
```

Or use the self-contained coverage script:

```bash
# Run tests and generate coverage reports with badge
python scripts/coverage_report.py --term
```

After running the tests with coverage, you can:

1. View the terminal report directly in your console
2. Open `htmlcov/index.html` in a browser to see the detailed HTML report
3. View the generated coverage badge (`coverage-badge.svg`)
4. Check the [Codecov dashboard](https://codecov.io/gh/copick/copick-torch) for the project's coverage metrics

## Code of Conduct

This project adheres to the Contributor Covenant [code of conduct](https://github.com/chanzuckerberg/.github/blob/main/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [opensource@chanzuckerberg.com](mailto:opensource@chanzuckerberg.com).

## Reporting Security Issues

If you believe you have found a security issue, please responsibly disclose by contacting us at [security@chanzuckerberg.com](mailto:security@chanzuckerberg.com).
