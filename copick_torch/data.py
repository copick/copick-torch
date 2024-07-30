from morphospaces.datasets import CopickDataset

def load_dataset(copick_config_path, run_names, tomo_type, user_id, session_id, segmentation_type, voxel_spacing, transform, patch_shape, patch_stride, patch_filter_key, patch_threshold, store_unique_label_values=True):
    return CopickDataset.from_copick_project(
        copick_config_path=copick_config_path,
        run_names=run_names.split(","),
        tomo_type=tomo_type,
        user_id=user_id,
        session_id=session_id,
        segmentation_type=segmentation_type,
        voxel_spacing=voxel_spacing,
        transform=transform,
        patch_shape=patch_shape,
        stride_shape=patch_stride,
        patch_filter_key=patch_filter_key,
        patch_threshold=patch_threshold,
        store_unique_label_values=store_unique_label_values,
    )
