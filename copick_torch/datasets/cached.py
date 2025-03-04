import logging
import pandas as pd
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple

from copick_torch.datasets.base import BaseDataset

logger = logging.getLogger(__name__)

class CachedParquetDataset(BaseDataset):
    """Dataset that loads pre-processed data from parquet files."""
    
    def __init__(
        self,
        parquet_path: str,
        name_to_pdb: Dict[str, str],
        box_size: int,
        augment: bool = False,
        device: str = "cpu",
        seed: Optional[int] = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None
    ):
        super().__init__(box_size, augment, device, seed, rank, world_size)
        self.parquet_path = parquet_path
        self.name_to_pdb = name_to_pdb
        
        # Initialize molecule mapping
        self._initialize_molecule_mapping()
        
        # Load data with distributed awareness
        self._load_data()
        
        # Synchronize processes after data loading
        if self.world_size and self.world_size > 1:
            dist.barrier()

    def _initialize_molecule_mapping(self):
        """Create mapping between molecule IDs and indices using PDB IDs."""
        try:
            # Load unique molecule IDs
            molecule_ids = pd.read_parquet(self.parquet_path, columns=['molecule_id'])['molecule_id'].unique()
            
            # Filter out background, None, and NaN values
            non_background_ids = [
                mid for mid in molecule_ids 
                if mid not in ('background', None) and pd.notna(mid)
            ]
            
            if self.rank == 0:
                logger.info(f"Total unique molecules: {len(molecule_ids)}")
                logger.info(f"After filtering background/None: {len(non_background_ids)}")
            
            # Convert names to PDB IDs
            pdb_ids = []
            for name in non_background_ids:
                if name in self.name_to_pdb:
                    pdb_id = self.name_to_pdb[name]
                    pdb_ids.append(pdb_id)
                else:
                    logger.warning(f"No PDB ID mapping found for molecule: {name}")
                    
            # Create bidirectional mappings
            self.molecule_to_idx = {pdb: idx for idx, pdb in enumerate(sorted(pdb_ids))}
            self.idx_to_molecule = {idx: pdb for pdb, idx in self.molecule_to_idx.items()}
                
        except Exception as e:
            logger.error(f"Error in _initialize_molecule_mapping: {str(e)}")
            raise

    def _load_data(self):
        """Load data with validation."""
        try:
            self.df = pd.read_parquet(self.parquet_path)
            if self.rank == 0:
                logger.info(f"Loaded {len(self.df)} samples")
                
            # Validate data
            missing_cols = set(['subvolume', 'shape', 'molecule_id']) - set(self.df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            def process_subvolume(x):
                if isinstance(x, bytes):
                    try:
                        return np.frombuffer(x, dtype=np.float32).copy()
                    except:
                        return None
                return x
                
            self.df['subvolume'] = self.df['subvolume'].apply(process_subvolume)
            
            # Filter out invalid rows
            valid_mask = self.df['subvolume'].notna()
            invalid_count = (~valid_mask).sum()
            if invalid_count > 0:
                logger.warning(f"Found {invalid_count} invalid subvolumes")
                self.df = self.df[valid_mask].reset_index(drop=True)
                
            if len(self.df) == 0:
                raise ValueError("No valid data found after filtering")
                
        except Exception as e:
            logger.error(f"Error loading data on rank {self.rank}: {e}")
            raise

    def __getitem__(self, idx):
        try:
            sample = self.df.iloc[idx]
            subvolume = sample['subvolume']
            
            if isinstance(subvolume, (bytes, np.ndarray)):
                if isinstance(subvolume, bytes):
                    subvolume = np.frombuffer(subvolume, dtype=np.float32)
                
                # Parse shape - could be stored as str or list
                shape = sample['shape']
                if isinstance(shape, str):
                    import json
                    shape = json.loads(shape)
                
                subvolume = subvolume.reshape(shape)
                
                if self.augment:
                    subvolume = self._augment_subvolume(subvolume)
                    
                subvolume = np.expand_dims(subvolume, axis=0)
                subvolume = torch.from_numpy(subvolume).to(dtype=torch.float32)
            else:
                raise ValueError(f"Unexpected subvolume type: {type(subvolume)}")
            
            # Special case for background
            name = sample['molecule_id']
            if name == 'background':
                molecule_idx = -1  # Use -1 to indicate background
            else:
                # Regular case - convert name to PDB ID then to index
                if name in self.name_to_pdb and self.name_to_pdb[name] in self.molecule_to_idx:
                    pdb_id = self.name_to_pdb[name]
                    molecule_idx = self.molecule_to_idx[pdb_id]
                else:
                    logger.warning(f"Unknown molecule {name}, treating as background")
                    molecule_idx = -1
            
            return subvolume, torch.tensor(molecule_idx, dtype=torch.long)
                
        except Exception as e:
            logger.error(f"Error loading item {idx}: {str(e)}")
            # Return a default value in case of error
            return torch.zeros((1, self.box_size, self.box_size, self.box_size), dtype=torch.float32), torch.tensor(-1, dtype=torch.long)

    def __len__(self):
        return len(self.df)