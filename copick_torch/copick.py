import os
import dask.array as da
import numpy as np
import zarr
import copick
from typing import Dict, List, Tuple, Union
from numpy.typing import ArrayLike
from torch.utils.data import Dataset, ConcatDataset

from morphospaces.datasets.utils import (
    FilterSliceBuilder,
    PatchManager,
    SliceBuilder,
)

class CopickDataset(Dataset):
    """
    Implementation of the copick dataset that loads both image zarr arrays and their corresponding mask zarr arrays into numpy arrays, 
    constructing a map-style dataset, such as {'zarr_tomogram': Array([...], dtype=np.float32), 'zarr_mask': Array([...], dtype=np.float32)}.
    """

    def __init__(
        self,
        zarr_data: dict,  # {'zarr_tomogram': zarr_array, 'zarr_mask': zarr_array}
        transform=None,
        patch_shape: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (96, 96, 96),
        stride_shape: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (24, 24, 24),
        patch_filter_ignore_index: Tuple[int, ...] = (),
        patch_filter_key: str = "zarr_mask",
        patch_threshold: float = 0.6,
        patch_slack_acceptance=0.01,
        store_unique_label_values: bool = False,
    ):
        
        self.zarr_data = zarr_data
        self.dataset_keys = list(zarr_data.keys())
        self.transform = transform
        self.patch_shape = patch_shape
        self.stride_shape = stride_shape
        self.patch_filter_ignore_index = patch_filter_ignore_index
        self.patch_filter_key = patch_filter_key
        self.patch_threshold = patch_threshold
        self.patch_slack_acceptance = patch_slack_acceptance
        self.store_unique_label_values = store_unique_label_values

        # self.data: Dict[str, ArrayLike] = {
        #     key: zarr_data[key].astype(np.float32) for key in self.dataset_keys
        # }
        self.data = {key: da.from_zarr(zarr_data[key]) for key in self.dataset_keys}        
        self._init_states()

    def _init_states(self):
        assert self.patch_filter_key in self.data, "patch_filter_key must be a dataset key"
        self.patches = PatchManager(
            data=self.data,
            patch_shape=self.patch_shape,
            stride_shape=self.stride_shape,
            patch_filter_ignore_index=self.patch_filter_ignore_index,
            patch_filter_key=self.patch_filter_key,
            patch_threshold=self.patch_threshold,
            patch_slack_acceptance=self.patch_slack_acceptance,
        )

        if self.store_unique_label_values:
            self.unique_label_values = self._get_unique_labels()
        else:
            self.unique_label_values = None

    def _get_unique_labels(self) -> List[int]:
        unique_labels = set()
        for slice_indices in self.patches.slices:
            array = self.data[self.patch_filter_key]
            data_patch = array[slice_indices].compute()
            label_values = np.unique(data_patch)
            unique_labels.update(label_values)
        return list(unique_labels)

    @property
    def patch_count(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Dict[str, ArrayLike]:
        if idx >= len(self):
            raise StopIteration
        slice_indices = self.patches.slices[idx]
        data_patch = {key: array[slice_indices].compute() for key, array in self.data.items()}
        if self.transform is not None:
            data_patch = self.transform(data_patch)
        return data_patch

    def __len__(self) -> int:
        return self.patch_count

    @classmethod
    def from_copick_project(
        cls,
        copick_config_path: str,
        run_names: List[str],
        tomo_type: str,
        user_id: str,
        session_id: str,
        segmentation_type: str,
        voxel_spacing: float,
        transform=None,
        patch_shape: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (96, 96, 96),
        stride_shape: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (24, 24, 24),
        patch_filter_ignore_index: Tuple[int, ...] = (),
        patch_filter_key: str = "zarr_mask",
        patch_threshold: float = 0.6,
        patch_slack_acceptance=0.01,
        store_unique_label_values: bool = False,
    ):
        root = copick.from_file(copick_config_path)
        datasets = []

        for run_name in run_names:
            run = root.get_run(run_name)
            if run is None:
                raise ValueError(f"Run with name '{run_name}' not found.")
            
            voxel_spacing_obj = run.get_voxel_spacing(voxel_spacing)
            if voxel_spacing_obj is None:
                raise ValueError(f"Voxel spacing '{voxel_spacing}' not found in run '{run_name}'.")

            tomogram = voxel_spacing_obj.get_tomogram(tomo_type)
            if tomogram is None:
                raise ValueError(f"Tomogram type '{tomo_type}' not found for voxel spacing '{voxel_spacing}'.")

            # image = zarr.open(tomogram.zarr(), mode='r')['0']
            image = zarr.open(tomogram.zarr(), mode='r')['0']
            
            seg = run.get_segmentations(user_id=user_id, session_id=session_id, name=segmentation_type, voxel_size=voxel_spacing)
            if len(seg) == 0:
                raise ValueError(f"No segmentations found for session '{session_id}' and segmentation type '{segmentation_type}'.")

            # segmentation = zarr.open(seg[0].zarr(), mode="r")['0']
            
            # zarr_data = {
            #     'zarr_tomogram': image[:],
            #     'zarr_mask': segmentation[:]
            # }

            segmentation = zarr.open(seg[0].zarr(), mode="r")['0']
        
            zarr_data = {
                'zarr_tomogram': image,
                'zarr_mask': segmentation
            }
            
            dataset = cls(
                zarr_data=zarr_data,
                transform=transform,
                patch_shape=patch_shape,
                stride_shape=stride_shape,
                patch_filter_ignore_index=patch_filter_ignore_index,
                patch_filter_key=patch_filter_key,
                patch_threshold=patch_threshold,
                patch_slack_acceptance=patch_slack_acceptance,
                store_unique_label_values=store_unique_label_values,
            )
            datasets.append(dataset)
        
        if store_unique_label_values:
            unique_label_values = set()
            for dataset in datasets:
                unique_label_values.update(dataset.unique_label_values)
            return ConcatDataset(datasets), list(unique_label_values)

        return ConcatDataset(datasets)

# Example usage:
# copick_dataset = CopickDataset.from_copick_project(copick_config_path='path/to/config', run_names=['run1', 'run2'], tomo_type='tomo_type', user_id='user_id', session_id='session_id', segmentation_type='segmentation_type')
