# Copick-Torch Documentation

This directory contains documentation and examples for the Copick-Torch library.

## Dataset Examples

### SimpleCopickDataset Examples

The [simple_dataset_examples](./simple_dataset_examples) directory contains visualizations of examples from each class in the dataset used by the `SimpleCopickDataset` class.

To generate these examples, run:

```bash
python scripts/generate_simple_dataset_docs.py
```

This will:
1. Create a SimpleCopickDataset instance
2. Extract one example from each class
3. Save visualizations showing both central slices and sum projections in orthogonal views
4. Generate a markdown file with all the visualizations

### Augmentation Examples

The [augmentation_examples](./augmentation_examples) directory contains visualizations of various augmentations applied to the dataset used in the `spliced_mixup_example.py` example.

To generate these examples, run:

```bash
python scripts/generate_augmentation_docs.py
```

This will:
1. Create a dataset similar to the one used in the example
2. Extract one example from each class
3. Apply various augmentations to each example
4. Save visualizations showing both central slices and sum projections in orthogonal views
5. Generate a markdown file with all the visualizations

The visualizations show how different augmentations affect the appearance of the data, which can be useful for understanding the effects of various augmentation parameters.
