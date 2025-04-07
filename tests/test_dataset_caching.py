
import unittest
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import pickle
from unittest.mock import patch, MagicMock
from datetime import datetime

from copick_torch import SimpleCopickDataset


class TestDatasetCaching(unittest.TestCase):
    """Test the caching functionality of SimpleCopickDataset."""
    
    def setUp(self):
        # Create a temporary directory for caching
        self.test_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.test_dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Mock config path
        self.mock_config_path = "test_config.json"
        
        # Parameters for testing
        self.boxsize = (16, 16, 16)
        self.voxel_spacing = 10.0
        
        # Create test data
        self.test_subvolumes = [np.ones(self.boxsize), np.zeros(self.boxsize)]
        self.test_molecule_ids = [0, 1]
        self.test_keys = ["class1", "class2"]
        self.test_is_background = [False, False]
    
    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)
    
    @patch('copick_torch.dataset.SimpleCopickDataset._load_data')
    def test_get_cache_path_pickle(self, mock_load_data):
        """Test the _get_cache_path method with pickle format."""
        dataset = SimpleCopickDataset(
            config_path=self.mock_config_path,
            boxsize=self.boxsize,
            cache_dir=self.cache_dir,
            cache_format="pickle"
        )
        
        # Get cache path
        cache_path = dataset._get_cache_path()
        
        # Expected path format
        expected_path = os.path.join(
            self.cache_dir,
            f"{self.mock_config_path}_16x16x16_10.0.pkl"
        )
        
        self.assertEqual(cache_path, expected_path)
    
    @patch('copick_torch.dataset.SimpleCopickDataset._load_data')
    def test_get_cache_path_parquet(self, mock_load_data):
        """Test the _get_cache_path method with parquet format."""
        dataset = SimpleCopickDataset(
            config_path=self.mock_config_path,
            boxsize=self.boxsize,
            cache_dir=self.cache_dir,
            cache_format="parquet",
            include_background=True
        )
        
        # Get cache path
        cache_path = dataset._get_cache_path()
        
        # Expected path format
        expected_path = os.path.join(
            self.cache_dir,
            f"{self.mock_config_path}_16x16x16_10.0_with_bg.parquet"
        )
        
        self.assertEqual(cache_path, expected_path)
    
    @patch('copick_torch.dataset.SimpleCopickDataset._load_data')
    def test_get_cache_path_with_copick_root(self, mock_load_data):
        """Test the _get_cache_path method with copick_root instead of config_path."""
        # Create a mock copick_root with dataset IDs
        mock_root = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.id = 123
        mock_root.datasets = [mock_dataset]
        
        dataset = SimpleCopickDataset(
            config_path=None,
            copick_root=mock_root,
            boxsize=self.boxsize,
            cache_dir=self.cache_dir
        )
        
        # Get cache path
        cache_path = dataset._get_cache_path()
        
        # Expected path format with dataset IDs
        expected_path = os.path.join(
            self.cache_dir,
            f"datasets_123_16x16x16_10.0.parquet"
        )
        
        self.assertEqual(cache_path, expected_path)
    
    @patch('copick_torch.dataset.SimpleCopickDataset._load_data')
    def test_save_load_pickle(self, mock_load_data):
        """Test saving and loading data with pickle format."""
        # Create dataset
        dataset = SimpleCopickDataset(
            config_path=self.mock_config_path,
            boxsize=self.boxsize,
            cache_dir=self.cache_dir,
            cache_format="pickle"
        )
        
        # Set test data
        dataset._subvolumes = self.test_subvolumes
        dataset._molecule_ids = self.test_molecule_ids
        dataset._keys = self.test_keys
        dataset._is_background = self.test_is_background
        
        # Get cache path
        cache_path = dataset._get_cache_path()
        
        # Save to pickle
        dataset._save_to_pickle(cache_path)
        
        # Verify file was created
        self.assertTrue(os.path.exists(cache_path))
        
        # Create a new dataset to load the saved data
        new_dataset = SimpleCopickDataset(
            config_path=self.mock_config_path,
            boxsize=self.boxsize,
            cache_dir=self.cache_dir,
            cache_format="pickle"
        )
        
        # Clear data
        new_dataset._subvolumes = []
        new_dataset._molecule_ids = []
        new_dataset._keys = []
        new_dataset._is_background = []
        
        # Load from pickle
        new_dataset._load_from_pickle(cache_path)
        
        # Verify data was loaded correctly
        self.assertEqual(len(new_dataset._subvolumes), len(self.test_subvolumes))
        np.testing.assert_array_equal(new_dataset._subvolumes[0], self.test_subvolumes[0])
        np.testing.assert_array_equal(new_dataset._molecule_ids, self.test_molecule_ids)
        self.assertEqual(new_dataset._keys, self.test_keys)
        self.assertEqual(new_dataset._is_background, self.test_is_background)
    
    @patch('copick_torch.dataset.SimpleCopickDataset._load_data')
    def test_save_load_parquet(self, mock_load_data):
        """Test saving and loading data with parquet format."""
        # Create dataset
        dataset = SimpleCopickDataset(
            config_path=self.mock_config_path,
            boxsize=self.boxsize,
            cache_dir=self.cache_dir,
            cache_format="parquet"
        )
        
        # Set test data
        dataset._subvolumes = np.array(self.test_subvolumes)
        dataset._molecule_ids = np.array(self.test_molecule_ids)
        dataset._keys = self.test_keys
        dataset._is_background = np.array(self.test_is_background)
        
        # Get cache path
        cache_path = dataset._get_cache_path()
        
        # Save to parquet
        dataset._save_to_parquet(cache_path)
        
        # Verify files were created
        self.assertTrue(os.path.exists(cache_path))
        self.assertTrue(os.path.exists(cache_path.replace('.parquet', '_metadata.parquet')))
        
        # Create a new dataset to load the saved data
        new_dataset = SimpleCopickDataset(
            config_path=self.mock_config_path,
            boxsize=self.boxsize,
            cache_dir=self.cache_dir,
            cache_format="parquet"
        )
        
        # Clear data
        new_dataset._subvolumes = []
        new_dataset._molecule_ids = []
        new_dataset._keys = []
        new_dataset._is_background = []
        
        # Load from parquet
        new_dataset._load_from_parquet(cache_path)
        
        # Verify data was loaded correctly
        self.assertEqual(len(new_dataset._subvolumes), len(self.test_subvolumes))
        self.assertEqual(len(new_dataset._molecule_ids), len(self.test_molecule_ids))
        
        # Compare values (allowing for floating point differences)
        np.testing.assert_array_almost_equal(
            new_dataset._subvolumes[0], 
            self.test_subvolumes[0]
        )
        np.testing.assert_array_equal(
            new_dataset._molecule_ids,
            self.test_molecule_ids
        )
    
    @patch('copick_torch.dataset.SimpleCopickDataset._load_data')
    def test_load_or_process_data_with_cache(self, mock_load_data):
        """Test the _load_or_process_data method with an existing cache file."""
        # Create and save a cache file
        cache_file = os.path.join(self.cache_dir, f"{self.mock_config_path}_16x16x16_10.0.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'subvolumes': self.test_subvolumes,
                'molecule_ids': self.test_molecule_ids,
                'keys': self.test_keys,
                'is_background': self.test_is_background
            }, f)
        
        # Create dataset with cache_dir
        dataset = SimpleCopickDataset(
            config_path=self.mock_config_path,
            boxsize=self.boxsize,
            cache_dir=self.cache_dir,
            cache_format="pickle"
        )
        
        # The _load_data method should not be called since cache exists
        mock_load_data.assert_not_called()
        
        # Verify data was loaded from cache
        self.assertEqual(len(dataset._subvolumes), len(self.test_subvolumes))
    
    @patch('copick_torch.dataset.SimpleCopickDataset._load_data')
    def test_load_or_process_data_without_cache(self, mock_load_data):
        """Test the _load_or_process_data method without an existing cache file."""
        # Create dataset with cache_dir but no existing cache file
        dataset = SimpleCopickDataset(
            config_path=self.mock_config_path,
            boxsize=self.boxsize,
            cache_dir=self.cache_dir
        )
        
        # The _load_data method should be called to process data
        mock_load_data.assert_called_once()
    
    @patch('copick_torch.dataset.SimpleCopickDataset._load_data')
    def test_load_or_process_data_no_cache_dir(self, mock_load_data):
        """Test the _load_or_process_data method with no cache_dir."""
        # Create dataset without cache_dir
        dataset = SimpleCopickDataset(
            config_path=self.mock_config_path,
            boxsize=self.boxsize,
            cache_dir=None
        )
        
        # The _load_data method should be called directly
        mock_load_data.assert_called_once()
    
    @patch('copick_torch.dataset.SimpleCopickDataset._load_data')
    def test_max_samples_limit(self, mock_load_data):
        """Test the max_samples limit during initialization."""
        # Create dataset with max_samples
        max_samples = 1
        
        # Simulate cache loading by overriding _load_from_pickle
        def mock_load_from_pickle(cache_path):
            dataset._subvolumes = np.array(self.test_subvolumes)
            dataset._molecule_ids = np.array(self.test_molecule_ids)
            dataset._keys = self.test_keys
            dataset._is_background = np.array(self.test_is_background)
        
        # Create the dataset
        with patch('copick_torch.dataset.SimpleCopickDataset._load_from_pickle', 
                  side_effect=mock_load_from_pickle):
            with patch('os.path.exists', return_value=True):  # Pretend cache exists
                dataset = SimpleCopickDataset(
                    config_path=self.mock_config_path,
                    boxsize=self.boxsize,
                    cache_dir=self.cache_dir,
                    cache_format="pickle",
                    max_samples=max_samples
                )
        
        # Verify max_samples was applied
        self.assertEqual(len(dataset._subvolumes), max_samples)


if __name__ == '__main__':
    unittest.main()
