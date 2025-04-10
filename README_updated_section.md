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