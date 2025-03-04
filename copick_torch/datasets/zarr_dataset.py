import os
import logging
import threading
import sqlite3
import shutil
import tempfile
from pathlib import Path
import numpy as np
import torch
import zarr
from torch.utils.data import Dataset
from typing import List, Optional, Tuple, Dict

from copick_torch.datasets.base import BaseDataset

logger = logging.getLogger(__name__)

def create_temporary_db_copy(original_db_path: str, rank: int) -> str:
    """Create a temporary copy of the database for this node/rank.
    
    Args:
        original_db_path: Path to the original database
        rank: Current process rank
        
    Returns:
        Path to the temporary database copy
    """
    try:
        # Create temp directory if it doesn't exist
        temp_dir = Path(tempfile.gettempdir()) / "copick_temp_dbs"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create unique filename for this rank
        db_name = Path(original_db_path).name
        temp_db_path = temp_dir / f"{db_name}.rank{rank}"
        
        # Copy database if it doesn't exist or is older than original
        if not temp_db_path.exists() or (
            temp_db_path.stat().st_mtime < Path(original_db_path).stat().st_mtime
        ):
            logger.info(f"Rank {rank}: Copying database to {temp_db_path}")
            shutil.copy2(original_db_path, temp_db_path)
            
        return str(temp_db_path)
        
    except Exception as e:
        logger.error(f"Rank {rank}: Error creating temporary database: {str(e)}")
        raise

def cleanup_temporary_db(temp_db_path: str):
    """Clean up temporary database file.
    
    Args:
        temp_db_path: Path to temporary database to remove
    """
    try:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
            logger.info(f"Removed temporary database: {temp_db_path}")
    except Exception as e:
        logger.warning(f"Error cleaning up temporary database {temp_db_path}: {str(e)}")

class DistributedZarrDataset(BaseDataset):
    """Memory-efficient dataset with optimized data delivery for distributed training.
    
    This version creates a temporary copy of the database for each node to prevent
    concurrent access issues in distributed training.
    """
    
    def __init__(
        self,
        db_path: str,
        zarr_path: str,
        structure_ids: List[str],
        box_size: int,
        min_cache_size: int = 32,
        augment: bool = False,
        device: str = "cpu",
        seed: Optional[int] = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        snr_level: float = 5.0
    ):
        super().__init__(box_size, augment, device, seed, rank, world_size)
        self.zarr_path = zarr_path
        self.structure_ids = structure_ids
        self.snr_level = snr_level
        self.min_cache_size = min_cache_size
        
        # Create temporary database copy for this rank
        self.temp_db_path = create_temporary_db_copy(db_path, self.rank)
        
        # Initialize data storage
        self.loaded_data = []
        self.current_index = 0
        self.data_lock = threading.Lock()
        
        # Initialize components
        self.zarr_store = zarr.open(self.zarr_path, mode='r')
        self.conn = sqlite3.connect(self.temp_db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Path collection and synchronization
        self.paths = []
        self.stop_event = threading.Event()
        self.data_ready = threading.Event()
        
        # Get initial paths with rank-specific subset
        self._get_initial_paths()
        
        # Start background thread for loading
        self.load_thread = threading.Thread(target=self._load_worker)
        self.load_thread.daemon = True
        self.load_thread.start()
        
        # Wait for initial data to be loaded
        logger.info(f"Rank {self.rank}: Waiting for initial data load...")
        if not self.data_ready.wait(timeout=600):  # 10 minute timeout
            raise RuntimeError(f"Rank {self.rank}: Initial data loading timed out")
        logger.info(f"Rank {self.rank}: Initial data loaded successfully")

    def _get_initial_paths(self):
        """Get paths to load, distributed across ranks"""
        cursor = self.conn.cursor()
        db_structure_ids = [sid.replace('_', '-') for sid in self.structure_ids]
        
        query = f"""
        SELECT DISTINCT hierarchy_path, protein_name, ROWID
        FROM subvolumes
        WHERE ABS(snr - ?) < 0.01
        AND protein_name IN ({','.join('?' * len(db_structure_ids))})
        """
        
        cursor.execute(query, (self.snr_level, *db_structure_ids))
        all_paths = cursor.fetchall()
        
        # Distribute paths across ranks
        for i, row in enumerate(all_paths):
            if i % self.world_size == self.rank:
                self.paths.append({
                    'mol_id': row['protein_name'],
                    'path': row['hierarchy_path']
                })
        
        logger.info(f"Rank {self.rank}: Assigned {len(self.paths)} paths out of {len(all_paths)} total")

    def _load_and_process_subvolume(self, path_info):
        """Load and process a single subvolume"""
        try:
            path_components = path_info['path'].split('/')
            current_group = self.zarr_store
            for component in path_components:
                current_group = current_group[component]
            
            # Load and process
            subvol = current_group[:]
            subvol = (subvol - np.mean(subvol)) / (np.std(subvol) + 1e-6)
            
            if self.augment:
                subvol = self._augment_subvolume(subvol)
                
            subvol = np.expand_dims(subvol, axis=0)
            tensor = torch.from_numpy(subvol).to(dtype=torch.float32)
            
            # Get molecule index
            mol_idx = self.structure_ids.index(path_info['mol_id'])
            mol_idx = torch.tensor(mol_idx, dtype=torch.long)
            
            return (tensor, mol_idx)
        except Exception as e:
            logger.error(f"Rank {self.rank}: Error processing subvolume: {str(e)}")
            return None

    def _load_worker(self):
        """Load data in background with improved logging"""
        current_idx = 0
        total_paths = len(self.paths)
        
        logger.info(f"Rank {self.rank}: Starting to load {total_paths} paths")
        
        while not self.stop_event.is_set() and current_idx < total_paths:
            path_info = self.paths[current_idx]
            subvol = self._load_and_process_subvolume(path_info)
            
            if subvol is not None:
                with self.data_lock:
                    self.loaded_data.append(subvol)
                    current_size = len(self.loaded_data)
                current_idx += 1
                
                if current_size >= self.min_cache_size and not self.data_ready.is_set():
                    logger.info(f"Rank {self.rank}: Minimum cache size reached ({current_size} items)")
                    self.data_ready.set()
                
                if current_idx % 10 == 0:  # More frequent progress updates
                    logger.info(f"Rank {self.rank}: Loaded {current_idx}/{total_paths} subvolumes")
            else:
                logger.warning(f"Rank {self.rank}: Failed to load subvolume at index {current_idx}")
                current_idx += 1  # Skip failed items
        
        logger.info(f"Rank {self.rank}: Completed loading {len(self.loaded_data)} subvolumes")

    def __len__(self):
        return max(len(self.paths), 10000) if self.paths else 10000

    def __getitem__(self, idx):
        """Get item with fallback mechanism"""
        if not self.loaded_data:
            raise RuntimeError(f"Rank {self.rank}: No data available")
        
        with self.data_lock:
            # Simple round-robin access to loaded data
            item = self.loaded_data[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.loaded_data)
            return item

    def __del__(self):
        """Cleanup resources including temporary database"""
        self.stop_event.set()
        if hasattr(self, 'conn'):
            self.conn.close()
        
        # Clean up temporary database
        if hasattr(self, 'temp_db_path'):
            cleanup_temporary_db(self.temp_db_path)