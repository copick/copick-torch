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

# SplicedMixup with Fourier augmentation visualization (visualization only)
uv run examples/spliced_mixup_fourier_example.py
```

## Features

### Augmentations

`copick-torch` includes various data augmentation techniques for 3D tomographic data:

- **MixupAugmentation**: Implements the mixup technique (Zhang et al., 2018) for 3D volumes, creating virtual training examples by mixing pairs of inputs and their labels with a random proportion.
- **FourierAugment3D**: Implements Fourier-based augmentation that operates in the frequency domain, including random frequency dropout, phase noise injection, and intensity scaling.

Example usage of Fourier augmentation:

```python
from copick_torch.augmentations import FourierAugment3D

# Create the augmenter
fourier_aug = FourierAugment3D(
    freq_mask_prob=0.3,        # Probability of masking frequency components
    phase_noise_std=0.1,       # Standard deviation of phase noise
    intensity_scaling_range=(0.8, 1.2)  # Range for random intensity scaling
)

# Apply to a 3D volume
augmented_volume = fourier_aug(volume)
```

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
