# copick-torch

[![codecov](https://codecov.io/gh/copick/copick-torch/branch/main/graph/badge.svg)](https://codecov.io/gh/copick/copick-torch)

Torch utilities for [copick](https://github.com/copick/copick)

## Dataset classes

- `SimpleCopickDataset`: Main dataset class with caching and augmentation support
- `MinimalCopickDataset`: Simpler dataset implementation with optional preloading

### MinimalCopickDataset Usage

#### Direct usage in Python

```python
from copick_torch import MinimalCopickDataset
from torch.utils.data import DataLoader

# Create a minimal dataset - no caching, no augmentation
dataset = MinimalCopickDataset(
    dataset_id=10440,                 # Dataset ID from CZ portal
    overlay_root='/tmp/test/',        # Overlay root directory
    boxsize=(48, 48, 48),             # Size of the subvolumes
    voxel_spacing=10.012,             # Voxel spacing
    include_background=True,          # Include background samples
    background_ratio=0.2,             # Background ratio
    min_background_distance=48,       # Minimum distance from particles for background
    max_samples=None                  # No limit on samples
)

# Print dataset information
print(f"Dataset size: {len(dataset)}")
print(f"Classes: {dataset.keys()}")
print(f"Class distribution: {dataset.get_class_distribution()}")

# Create a DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Training loop
for volume, label in dataloader:
    # volume shape: [batch_size, 1, depth, height, width]
    # label: [batch_size] class indices
    # Your training code here
    pass
```

#### Saving and loading datasets

The `MinimalCopickDataset` supports preloading all subvolumes into memory and saving the actual tensor data to disk, making it easy to share and load datasets without needing access to the original tomogram data:

```python
from copick_torch import MinimalCopickDataset

# Create a dataset with preloading enabled (default)
dataset = MinimalCopickDataset(
    dataset_id=10440,
    overlay_root='/tmp/copick_overlay',
    preload=True  # This preloads all subvolumes into memory
)

# Save the dataset with preloaded tensors
dataset.save('/path/to/save')

# Load the dataset from the saved tensors (no need for original tomogram data)
loaded_dataset = MinimalCopickDataset.load('/path/to/save')
```

You can also use the provided utility script to save a dataset directly from the command line:

```bash
# Save with preloading (default)
python scripts/save_torch_dataset.py --dataset_id 10440 --output_dir /path/to/save

# Save without preloading (not recommended)
python scripts/save_torch_dataset.py --dataset_id 10440 --output_dir /path/to/save --no-preload
```

Options:
```
  --dataset_id DATASET_ID   Dataset ID from the CZ cryoET Data Portal
  --output_dir OUTPUT_DIR   Directory to save the dataset
  --overlay_root OVERLAY_ROOT
                            Root directory for overlay storage (default: /tmp/copick_overlay)
  --boxsize Z Y X           Size of subvolumes to extract (default: 48 48 48)
  --voxel_spacing SPACING   Voxel spacing to use (default: 10.012)
  --include_background      Include background samples in the dataset
  --background_ratio RATIO  Ratio of background to particle samples (default: 0.2)
  --no-preload              Disable preloading tensors (not recommended)
  --verbose                 Enable verbose output
```

#### Inspecting saved datasets

You can display detailed information about a saved dataset using the provided utility script:

```bash
python scripts/info_torch_dataset.py --input_dir /path/to/saved/dataset
```

This will display:
- Basic dataset metadata (dataset ID, box size, voxel spacing, etc.)
- Class mapping information
- Total number of samples
- Class distribution (counts and percentages)
- Tomogram information
- Sample volume shape

The script can also generate visualizations:

```bash
python scripts/info_torch_dataset.py --input_dir /path/to/dataset --output_pdf dataset_report.pdf --samples_per_class 5
```

Options:
```
  --input_dir INPUT_DIR     Directory where the dataset is saved
  --output_pdf OUTPUT_PDF   Path to save visualization PDF (default: input_dir/dataset_overview.pdf)
  --samples_per_class SAMPLES_PER_CLASS
                            Number of sample visualizations per class (default: 3)
  --verbose                 Enable verbose output
```

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

# Generate dataset documentation
python scripts/generate_dataset_examples.py

# Save dataset to disk with preloaded tensors
python scripts/save_torch_dataset.py --dataset_id 10440 --output_dir /path/to/save

# Display information about a saved dataset
python scripts/info_torch_dataset.py --input_dir /path/to/save

# Visualize dataset with orthogonal views and projections
python examples/visualize_dataset.py --dataset_dir /path/to/save --output_file report.png

# Create enhanced visual report with sum projections
python examples/visualize_dataset_enhanced.py --dataset_dir /path/to/save --output_file report_enhanced.png
```

## Dataset Visualization

The repository includes two scripts for visualizing datasets:

### Basic Visualization

The `visualize_dataset.py` script creates a simple visualization of dataset samples with orthogonal views and maximum intensity projections:

```bash
python examples/visualize_dataset.py --dataset_dir /path/to/saved/dataset --output_file report.png
```

Options:
```
  --dataset_dir DATASET_DIR   Directory where the dataset was saved
  --output_file OUTPUT_FILE   Output file for the visualization (default: dataset_visualization.png)
  --samples_per_class SAMPLES_PER_CLASS
                            Number of samples to display per class (default: 2)
  --dpi DPI                 DPI for the output image (default: 150)
  --verbose                 Enable verbose output
```

### Enhanced Visualization

The `visualize_dataset_enhanced.py` script creates a more elegant visualization with sum projections and better layout:

```bash
python examples/visualize_dataset_enhanced.py --dataset_dir /path/to/saved/dataset --output_file report_enhanced.png
```

Options:
```
  --dataset_dir DATASET_DIR   Directory where the dataset was saved
  --output_file OUTPUT_FILE   Output file for the visualization (default: dataset_visualization_enhanced.png)
  --samples_per_class SAMPLES_PER_CLASS
                            Number of samples to display per class (default: 2)
  --dpi DPI                 DPI for the output image (default: 150)
  --cmap CMAP               Colormap to use for visualization (default: viridis)
  --verbose                 Enable verbose output
```

## Features

### Augmentations

`copick-torch` includes various MONAI-based data augmentation techniques for 3D tomographic data:

- **MixupTransform**: MONAI-compatible implementation of the Mixup technique (Zhang et al., 2018), creating virtual training examples by mixing pairs of inputs and their labels with a random proportion.
- **FourierAugment3D**: MONAI-compatible implementation of Fourier-based augmentation that operates in the frequency domain, including random frequency dropout, phase noise injection, and intensity scaling.

Example usage of MONAI-based Fourier augmentation:

```python
from copick_torch.monai_augmentations import FourierAugment3D

# Create the augmenter
fourier_aug = FourierAugment3D(
    freq_mask_prob=0.3,        # Probability of masking frequency components
    phase_noise_std=0.1,       # Standard deviation of phase noise
    intensity_scaling_range=(0.8, 1.2),  # Range for random intensity scaling
    prob=1.0                   # Probability of applying the transform
)

# Apply to a 3D volume (with PyTorch tensor)
augmented_volume = fourier_aug(volume_tensor)
```

### Documentation

See the [docs directory](./docs) for documentation and examples:

- [Augmentation Examples](./docs/augmentation_examples): Visualizations of various augmentations applied to different classes from the dataset used in the `spliced_mixup_example.py` example.
- [Dataset Examples](./docs/dataset_examples): Examples of volumes from each class in the dataset used by the CopickDataset classes.

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
