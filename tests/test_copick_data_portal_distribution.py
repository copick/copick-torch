import unittest
import os
import tempfile
import shutil
import json
import numpy as np
from collections import Counter

import copick
from copick_torch import SimpleCopickDataset


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
        cls.cache_dir = os.path.join(cls.temp_dir, 'cache')
        os.makedirs(cls.cache_dir, exist_ok=True)
        
        # Define the test dataset ID from the CryoET Data Portal
        cls.dataset_id = 10440
        cls.overlay_root = "./overlay"
        
        # Create a temporary config file for the dataset
        cls.config_path = os.path.join(cls.temp_dir, "test_config.json")
        
        # Create the config file content
        config = {
            "config_type": "cryoet_data_portal",
            "name": "Test Dataset",
            "description": "Test Dataset for distribution verification",
            "version": "1.0.0",
            "overlay_root": cls.overlay_root,
            "overlay_fs_args": {
                "auto_mkdir": True
            },
            "dataset_ids": [cls.dataset_id]
        }
        
        # Write the config to file
        with open(cls.config_path, 'w') as f:
            json.dump(config, f)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.temp_dir)
    
    def test_pickable_object_distribution(self):
        """
        Test that the distribution of pickable objects in SimpleCopickDataset
        matches the distribution in the CryoET Data Portal.
        """
        # First, get the actual picks from Copick directly
        try:
            # Load the Copick project
            project = copick.from_file(self.config_path)
            
            # Create a counter to track object counts directly from Copick
            copick_object_counts = Counter()
            
            # Loop through all runs and count the pickable objects
            for run in project.runs:
                for picks in run.get_picks():
                    if picks.from_tool:
                        # Get the object name and count the points
                        object_name = picks.pickable_object_name
                        points, _ = picks.numpy()
                        copick_object_counts[object_name] += len(points)
            
            # Now, create the SimpleCopickDataset with the same config
            dataset = SimpleCopickDataset(
                config_path=self.config_path,
                boxsize=(32, 32, 32),
                voxel_spacing=10.0,
                cache_dir=None  # Don't use caching for this test
            )
            
            # Get the distribution of classes in the dataset
            dataset_distribution = dataset.get_class_distribution()
            
            # Check that all pickable objects are represented in the dataset
            for object_name, count in copick_object_counts.items():
                self.assertIn(object_name, dataset_distribution, 
                              f"Object {object_name} is missing from the dataset")
                
                # Calculate the proportion of each object in both distributions
                copick_proportion = count / sum(copick_object_counts.values())
                dataset_proportion = dataset_distribution[object_name] / sum(dataset_distribution.values())
                
                # Assert that the proportion is similar (within 5% margin)
                diff = abs(copick_proportion - dataset_proportion)
                self.assertLess(diff, 0.05, 
                                f"Proportion mismatch for {object_name}: "
                                f"Copick: {copick_proportion:.3f}, Dataset: {dataset_proportion:.3f}")
                
            # Print the distributions for logging purposes
            print("\nCopick Object Counts:")
            for obj, count in copick_object_counts.items():
                print(f"  {obj}: {count}")
                
            print("\nDataset Distribution:")
            for obj, count in dataset_distribution.items():
                print(f"  {obj}: {count}")
                
        except Exception as e:
            self.fail(f"Test failed with error: {str(e)}")


if __name__ == '__main__':
    unittest.main()
