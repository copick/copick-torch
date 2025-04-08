import unittest
import os
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock
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
        
        # Create the config file content with multiple pickable objects
        config = {
            "config_type": "cryoet_data_portal",
            "name": "Test Dataset",
            "description": "Test Dataset for distribution verification",
            "version": "1.0.0",
            "overlay_root": cls.overlay_root,
            "overlay_fs_args": {
                "auto_mkdir": True
            },
            "dataset_ids": [cls.dataset_id],
            "pickable_objects": [
                {
                    "name": "cytosolic-ribosome",
                    "go_id": "GO:0022626",
                    "is_particle": True,
                    "label": 1,
                    "color": [0, 255, 0, 255],
                    "radius": 50.0
                },
                {
                    "name": "beta-amylase",
                    "go_id": "UniProtKB:P10537",
                    "is_particle": True,
                    "label": 2,
                    "color": [255, 0, 255, 255],
                    "radius": 50.0
                },
                {
                    "name": "thyroglobulin",
                    "go_id": "UniProtKB:P01267",
                    "is_particle": True,
                    "label": 3,
                    "color": [0, 127, 255, 255],
                    "radius": 50.0
                },
                {
                    "name": "virus-like-capsid",
                    "go_id": "GO:0170047",
                    "is_particle": True,
                    "label": 4,
                    "color": [255, 127, 0, 255],
                    "radius": 50.0
                },
                {
                    "name": "ferritin-complex",
                    "go_id": "GO:0070288",
                    "is_particle": True,
                    "label": 5,
                    "color": [127, 191, 127, 255],
                    "radius": 50.0
                },
                {
                    "name": "beta-galactosidase",
                    "go_id": "UniProtKB:P00722",
                    "is_particle": True,
                    "label": 6,
                    "color": [94, 6, 164, 255],
                    "radius": 50.0
                }
            ]
        }
        
        # Write the config to file
        with open(cls.config_path, 'w') as f:
            json.dump(config, f)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.temp_dir)
    
    @patch('copick_torch.dataset.SimpleCopickDataset._extract_subvolume_with_validation')
    @patch('copick.Run.get_picks')
    @patch('copick.from_file')
    def test_pickable_object_distribution(self, mock_from_file, mock_get_picks, mock_extract_subvolume):
        """
        Test that the distribution of pickable objects in SimpleCopickDataset
        matches the distribution in the CryoET Data Portal.
        
        This test mocks the zarr fetching to avoid the slow download while still
        testing the object distribution consistency.
        """
        # Define mock data with consistent object distributions
        object_names = ["cytosolic-ribosome", "beta-amylase", "thyroglobulin", 
                        "virus-like-capsid", "ferritin-complex", "beta-galactosidase"]
        
        # Define points for each object with somewhat random-looking coordinates
        object_points = {
            "cytosolic-ribosome": np.array([[100, 200, 30], [150, 250, 35], [200, 300, 40], [250, 350, 45], [300, 400, 50]]),
            "beta-amylase": np.array([[500, 600, 70], [550, 650, 75], [600, 700, 80]]),
            "thyroglobulin": np.array([[800, 900, 110], [850, 950, 115], [900, 1000, 120], [950, 1050, 125]]),
            "virus-like-capsid": np.array([[1100, 1200, 150], [1150, 1250, 155]]),
            "ferritin-complex": np.array([[1300, 1400, 180], [1350, 1450, 185], [1400, 1500, 190]]),
            "beta-galactosidase": np.array([[1600, 1700, 210], [1650, 1750, 215], [1700, 1800, 220]])
        }
        
        # Setup mock picks and run
        mock_picks = []
        for object_name, points in object_points.items():
            mock_pick = MagicMock()
            mock_pick.pickable_object_name = object_name
            mock_pick.from_tool = True
            mock_pick.numpy.return_value = (points, None)
            mock_picks.append(mock_pick)
        
        # Setup mock run
        mock_run = MagicMock()
        mock_run.name = "mock_run_16463"
        mock_run.get_picks.return_value = mock_picks
        
        # Setup mock voxel_spacing
        mock_vs = MagicMock()
        
        # Mock the tomogram to return a zeros array
        mock_tomogram = MagicMock()
        mock_tomogram.numpy.return_value = np.zeros((200, 200, 200))
        mock_vs.tomograms = [mock_tomogram]
        mock_run.get_voxel_spacing.return_value = mock_vs
        
        # Setup mock project
        mock_project = MagicMock()
        mock_project.runs = [mock_run]
        mock_from_file.return_value = mock_project
        
        # Mock the extract_subvolume method to always return a valid subvolume
        mock_extract_subvolume.return_value = (np.zeros((32, 32, 32)), True, "valid")
        
        # First, get the actual pick counts directly from our mocked Copick
        project = copick.from_file(self.config_path)
        run = project.runs[0]
        
        # Create a counter to track object counts directly from Copick
        copick_object_counts = Counter()
        
        # Count the pickable objects in the mocked run
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
            voxel_spacing=10.012,
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


if __name__ == '__main__':
    unittest.main()
