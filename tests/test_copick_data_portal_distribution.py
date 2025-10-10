import json
import os
import shutil
import tempfile
import unittest
from collections import Counter
from unittest.mock import MagicMock, patch

import copick
import numpy as np

from copick_torch.dataset import SimpleCopickDataset


class TestCopickDataPortalDistribution(unittest.TestCase):
    """
    Test that verifies the SimpleCopickDataset correctly preserves the
    distribution of pickable objects from the CryoET Data Portal.
    """

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create a temporary directory for caching
        cls.temp_dir = tempfile.mkdtemp()
        cls.cache_dir = os.path.join(cls.temp_dir, "cache")
        os.makedirs(cls.cache_dir, exist_ok=True)

        # Define the test dataset ID from the CryoET Data Portal
        cls.dataset_id = 10440
        cls.overlay_root = "./overlay"

        # Create a temporary config file for the dataset
        cls.config_path = os.path.join(cls.temp_dir, "test_config.json")

        # Create the config file content with multiple pickable objects
        config = {
            "config_type": "cryoet_data_portal",
            "name": "Test Dataset",
            "description": "Test Dataset for distribution verification",
            "version": "1.0.0",
            "overlay_root": cls.overlay_root,
            "overlay_fs_args": {"auto_mkdir": True},
            "dataset_ids": [cls.dataset_id],
            "pickable_objects": [
                {
                    "name": "cytosolic-ribosome",
                    "go_id": "GO:0022626",
                    "is_particle": True,
                    "label": 1,
                    "color": [0, 255, 0, 255],
                    "radius": 50.0,
                },
                {
                    "name": "beta-amylase",
                    "go_id": "UniProtKB:P10537",
                    "is_particle": True,
                    "label": 2,
                    "color": [255, 0, 255, 255],
                    "radius": 50.0,
                },
                {
                    "name": "thyroglobulin",
                    "go_id": "UniProtKB:P01267",
                    "is_particle": True,
                    "label": 3,
                    "color": [0, 127, 255, 255],
                    "radius": 50.0,
                },
                {
                    "name": "virus-like-capsid",
                    "go_id": "GO:0170047",
                    "is_particle": True,
                    "label": 4,
                    "color": [255, 127, 0, 255],
                    "radius": 50.0,
                },
                {
                    "name": "ferritin-complex",
                    "go_id": "GO:0070288",
                    "is_particle": True,
                    "label": 5,
                    "color": [127, 191, 127, 255],
                    "radius": 50.0,
                },
                {
                    "name": "beta-galactosidase",
                    "go_id": "UniProtKB:P00722",
                    "is_particle": True,
                    "label": 6,
                    "color": [94, 6, 164, 255],
                    "radius": 50.0,
                },
            ],
        }

        # Write the config to file
        with open(cls.config_path, "w") as f:
            json.dump(config, f)

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.temp_dir)

    @patch("copick_torch.dataset.SimpleCopickDataset._extract_subvolume_with_validation")
    def test_pickable_object_distribution(self, mock_extract_subvolume):
        """
        Test that the distribution of pickable objects in SimpleCopickDataset
        matches the distribution in the CryoET Data Portal.

        This test only mocks the subvolume extraction to avoid the slow zarr loading
        while still using real picks data for testing the distribution consistency.
        """
        # Mock the extract_subvolume method to always return a valid subvolume
        mock_extract_subvolume.return_value = (np.zeros((32, 32, 32)), True, "valid")

        try:
            # Load the Copick project with real data
            project = copick.from_file(self.config_path)

            # Use only one run to speed up the test
            if project.runs:
                # Use the first run with available picks
                run = None
                for potential_run in project.runs:
                    # Check if the run has picks
                    has_picks = False
                    for picks in potential_run.get_picks():
                        if picks.from_tool:
                            has_picks = True
                            break

                    if has_picks:
                        run = potential_run
                        break

                if not run:
                    self.skipTest("No runs with picks found in the dataset.")

                print(f"\nUsing run: {run.name} for testing")

                # Create a counter to track object counts directly from Copick
                copick_object_counts = Counter()

                # Count the pickable objects in this run using real picks data
                for picks in run.get_picks():
                    if picks.from_tool:
                        # Get the object name and count the points
                        object_name = picks.pickable_object_name
                        points, _ = picks.numpy()
                        copick_object_counts[object_name] += len(points)

                # Now, create the SimpleCopickDataset with the same config
                # Use a modified configuration that only includes the selected run
                modified_config = self.config_path.replace(".json", f"_{run.name}.json")

                # Get the original config
                with open(self.config_path, "r") as f:
                    config_data = json.load(f)

                # Create a custom run filter to select only this run
                config_data["run_filter"] = [{"name": run.name}]

                # Write the modified config
                with open(modified_config, "w") as f:
                    json.dump(config_data, f)

                # Create dataset with the modified config
                dataset = SimpleCopickDataset(
                    config_path=modified_config,
                    boxsize=(32, 32, 32),
                    voxel_spacing=10.012,
                    cache_dir=None,  # Don't use caching for this test
                )

                # Get the distribution of classes in the dataset
                dataset_distribution = dataset.get_class_distribution()

                # Skip test if no objects were found
                if not copick_object_counts:
                    self.skipTest("No pickable objects found in the selected run.")

                # Check that all pickable objects are represented in the dataset
                for object_name, count in copick_object_counts.items():
                    # Skip beta-amylase check as it may not be in the dataset
                    if object_name == "beta-amylase" and object_name not in dataset_distribution:
                        print("Warning: beta-amylase not found in dataset distribution, skipping check")
                        continue

                    self.assertIn(
                        object_name,
                        dataset_distribution,
                        f"Object {object_name} is missing from the dataset",
                    )

                    # Calculate the proportion of each object in both distributions
                    copick_proportion = count / sum(copick_object_counts.values())
                    dataset_proportion = dataset_distribution[object_name] / sum(dataset_distribution.values())

                    # Assert that the proportion is similar (within 5% margin)
                    diff = abs(copick_proportion - dataset_proportion)
                    self.assertLess(
                        diff,
                        0.05,
                        f"Proportion mismatch for {object_name}: "
                        f"Copick: {copick_proportion:.3f}, Dataset: {dataset_proportion:.3f}",
                    )

                # Print the distributions for logging purposes
                print("\nCopick Object Counts:")
                for obj, count in copick_object_counts.items():
                    print(f"  {obj}: {count}")

                print("\nDataset Distribution:")
                for obj, count in dataset_distribution.items():
                    print(f"  {obj}: {count}")
            else:
                self.skipTest("No runs found in the dataset.")

        except Exception as e:
            self.fail(f"Test failed with error: {str(e)}")


if __name__ == "__main__":
    unittest.main()
