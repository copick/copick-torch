import logging
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class BaseDataset(Dataset):
    """Base dataset class with common functionality."""
    
    def __init__(
        self,
        box_size: int,
        augment: bool = False,
        device: str = "cpu",
        seed: Optional[int] = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        self.box_size = box_size
        self.augment = augment
        self.device = device
        self.seed = seed
        self.rank = rank if rank is not None else 0
        self.world_size = world_size if world_size is not None else 1
        
        self._set_random_seed()
        
    def _set_random_seed(self):
        """Set random seed with distributed awareness."""
        if self.seed is not None:
            effective_seed = self.seed + (self.rank or 0)
            np.random.seed(effective_seed)
            torch.manual_seed(effective_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(effective_seed)
                
    def _augment_subvolume(self, subvolume: np.ndarray) -> np.ndarray:
        """Apply data augmentation to 3D volume."""
        if not self.augment:
            return subvolume
            
        # Make a contiguous copy to avoid stride issues
        subvolume = np.ascontiguousarray(subvolume)
        
        # Random rotation
        if np.random.random() > 0.5:
            k = np.random.randint(1, 4)
            axes = tuple(np.random.choice([0, 1, 2], size=2, replace=False))
            subvolume = np.rot90(subvolume, k=k, axes=axes)
            
        # Random flips
        for axis in range(3):
            if np.random.random() > 0.5:
                subvolume = np.flip(subvolume, axis=axis).copy()
                
        return subvolume

    def __len__(self):
        raise NotImplementedError("Subclasses must implement __len__")

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement __getitem__")