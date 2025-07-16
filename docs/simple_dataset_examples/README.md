# SimpleCopickDataset Examples

This directory will contain visualizations of examples from each class in the dataset used by the `SimpleCopickDataset` class.

To generate these examples, run:

```bash
python scripts/generate_simple_dataset_docs.py
```

This will:
1. Create a SimpleCopickDataset instance
2. Extract one example from each class
3. Save visualizations showing both central slices and sum projections in orthogonal views
4. Generate a complete version of this markdown file with all the visualizations

## Example Visualization

The visualizations will show cross-sections and projections of the 3D volumes for each class in the dataset. For each class, you'll see:

- Top row: Central slices in XY, XZ, and YZ planes
- Bottom row: Sum projections in XY, XZ, and YZ planes

This helps to understand the 3D structure of the data from different views.

## Dataset Information

The dataset used in this example is created using the `SimpleCopickDataset` class with the following parameters:

```python
dataset = SimpleCopickDataset(
    dataset_id=10440,          # Experimental dataset ID
    overlay_root='/tmp/test/', # Overlay root directory
    boxsize=(48, 48, 48),      # Size of the subvolumes
    augment=False,             # Disable augmentations for examples
    cache_dir='./cache',       # Cache directory
    cache_format='parquet',    # Cache format
    voxel_spacing=10.012,      # Voxel spacing
    include_background=True,   # Include background samples
    background_ratio=0.2,      # Background ratio
    min_background_distance=48,# Minimum distance from particles for background
    max_samples=100            # Maximum number of samples to generate
)
```

Once generated, this file will also include:
- One image per class in the dataset
- A table showing the class distribution
- A usage example for the `SimpleCopickDataset` class
