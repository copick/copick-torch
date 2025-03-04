import os
import zarr
import pytest
import numpy as np
import json
from pathlib import Path
from torch.utils.data import ConcatDataset, DataLoader
from copick import from_file
from copick_torch.copick import CopickDataset  # Adjust the import path if necessary


# Helper function to create a copick config JSON
def create_copick_config_json(tmp_path: Path, config_filename: str):
    copick_config = {
        "config_type": "cryoet_data_portal",
        "name": "Example Project",
        "description": "This is an example project.",
        "version": "0.5.0",
        "pickable_objects": [
            {
                "name": "ribosome",
                "is_particle": True,
                "go_id": "GO:0022626",
                "label": 1,
                "color": [0, 117, 220, 255],
                "radius": 150
            },
            {
                "name": "atpase",
                "is_particle": True,
                "go_id": "GO:0045259",
                "label": 2,
                "color": [251, 192, 147, 255],
                "radius": 150
            },
            {
                "name": "membrane",
                "is_particle": False,
                "go_id": "GO:0016020",
                "label": 3,
                "color": [200, 200, 200, 255],
                "radius": 10
            }
        ],
        "overlay_root": str(tmp_path / "random_points"),
        "overlay_fs_args": {
            "auto_mkdir": True
        },
        "dataset_ids": [10301]
    }

    config_path = tmp_path / config_filename
    with open(config_path, 'w') as f:
        json.dump(copick_config, f)

    return config_path



# Test the from_copick_project method and ensure Torch compatibility
def test_from_copick_project(tmp_path):
    # Step 1: Create copick config file
    config_filename = "config.json"
    config_path = create_copick_config_json(tmp_path, config_filename)

    # Step 2: Load the copick project
    root = from_file(config_path)

    print(f"Num runs: {len(root.runs)}")
    assert len(root.runs) == 18

    # Step 3: Select an existing run    
    run_name = root.runs[0].meta.name
    print(f"Using run: {run_name}")

    # Step 4: Use the from_copick_project method to load the dataset
    dataset, unique_labels = CopickDataset.from_copick_project(
        copick_config_path=config_path,
        run_names=[run_name],
        tomo_type="wbp",
        user_id="data-portal",
        session_id="67977",
        segmentation_type="membrane",
        voxel_spacing=7.84,
        store_unique_label_values=True,
    )

    # Step 5: Assert the dataset works with PyTorch's DataLoader
    assert isinstance(dataset, ConcatDataset), "The dataset is not a ConcatDataset"
    assert len(dataset) > 0, "The dataset is empty"
    
    # Check that unique labels are returned and correct (example label set, adjust as needed)
    assert isinstance(unique_labels, list), "Unique labels are not in list format"
    print(f"Unique labels: {unique_labels}")

    # Step 6: Create a DataLoader and ensure it works with PyTorch
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch in dataloader:
        assert isinstance(batch, dict), "Batch is not a dictionary"
        assert 'zarr_tomogram' in batch, "Missing 'zarr_tomogram' in batch"
        assert 'zarr_mask' in batch, "Missing 'zarr_mask' in batch"

        # Check the data shapes (example check, adjust according to your actual data shapes)
        assert batch['zarr_tomogram'].shape[0] == 2, "Batch size does not match"
        assert batch['zarr_mask'].shape[0] == 2, "Batch size does not match"
        
        print(f"Batch tomogram shape: {batch['zarr_tomogram'].shape}")
        print(f"Batch mask shape: {batch['zarr_mask'].shape}")

        # Stop after first batch for testing purposes
        break
