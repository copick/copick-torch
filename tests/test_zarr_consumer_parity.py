from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import zarr
from zarr.codecs import BloscCodec, GzipCodec, Shuffle, ZstdCodec
from zarr.storage import LocalStore

from copick_torch.copick import CopickDataset
from copick_torch.dataset import SimpleCopickDataset, SplicedMixupDataset
from copick_torch.minimal_dataset import MinimalCopickDataset
from copick_torch.nnunet.prepare import load_segmentation, load_volume
from copick_torch.storage import get_level_array

V3_LAYOUTS = {
    "unsharded-default-keys": {
        "chunks": (2, 3, 5),
        "compressors": None,
    },
    "alternate-chunks-gzip": {
        "chunks": (4, 3, 2),
        "compressors": (GzipCodec(level=1),),
    },
    "unsharded-v2-keys-blosc": {
        "chunks": (2, 3, 5),
        "compressors": (BloscCodec(cname="lz4", clevel=1),),
        "chunk_key_encoding": {"name": "v2", "separator": "/"},
    },
    "multishard-shuffle-zstd": {
        "chunks": (1, 3, 5),
        "shards": (4, 6, 10),
        "compressors": (Shuffle(elementsize=2), ZstdCodec(level=1)),
    },
}


class StoredEntity:
    def __init__(self, store):
        self._store = store
        self.tomo_type = "wbp-denoised"
        self.voxel_size = 10.0
        self.meta = SimpleNamespace(name="particle")

    def zarr(self):
        return self._store

    def numpy(self):
        return get_level_array(self)[:]


def write_store(path, layout):
    values = np.arange(8 * 9 * 10, dtype=np.int16).reshape(8, 9, 10)
    store = LocalStore(path)
    if layout == "legacy-v2-numeric":
        group = zarr.group(store=store, zarr_format=2)
        group.create_array("0", data=values, chunks=(2, 3, 5))
        group.attrs["multiscales"] = [
            {
                "version": "0.4",
                "axes": [{"name": axis, "type": "space", "unit": "angstrom"} for axis in ("z", "y", "x")],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [{"type": "scale", "scale": [10.0, 10.0, 10.0]}],
                    },
                ],
            },
        ]
    else:
        group = zarr.group(store=store, zarr_format=3)
        group.create_array("s0", data=values, dimension_names=("z", "y", "x"), **V3_LAYOUTS[layout])
        group.attrs["ome"] = {
            "version": "0.5",
            "multiscales": [
                {
                    "axes": [{"name": axis, "type": "space", "unit": "angstrom"} for axis in ("z", "y", "x")],
                    "datasets": [
                        {
                            "path": "s0",
                            "coordinateTransformations": [{"type": "scale", "scale": [10.0, 10.0, 10.0]}],
                        },
                    ],
                },
            ],
        }
    return StoredEntity(store), values


def project_for(entity):
    picks = SimpleNamespace(
        from_tool=True,
        pickable_object_name="particle",
        numpy=lambda: (np.array([[50.0, 40.0, 30.0]]), None),
    )
    voxel_spacing = SimpleNamespace(tomograms=[entity])
    run = SimpleNamespace(
        name="run-1",
        get_voxel_spacing=lambda _voxel_size: voxel_spacing,
        get_picks=lambda: [picks],
    )
    return SimpleNamespace(
        runs=[run],
        pickable_objects=[SimpleNamespace(name="particle", label=1)],
    )


def snapshot(path: Path):
    return {item.relative_to(path).as_posix(): item.read_bytes() for item in path.rglob("*") if item.is_file()}


def consume(entity):
    root = project_for(entity)
    copick_dataset = CopickDataset(copick_root=root, boxsize=(4, 4, 4), voxel_spacing=10.0, seed=1)
    simple_dataset = SimpleCopickDataset(copick_root=root, boxsize=(4, 4, 4), voxel_spacing=10.0, seed=1)
    minimal_dataset = MinimalCopickDataset(proj=root, boxsize=(4, 4, 4), voxel_spacing=10.0, preload=False)

    spliced_dataset = SplicedMixupDataset.__new__(SplicedMixupDataset)
    spliced_dataset.voxel_spacing = 10.0
    spliced_dataset.exp_root = object()
    spliced_dataset.synth_root = object()
    spliced_dataset._synth_mask_data = {}
    spliced_dataset._get_available_tomograms = lambda _root, _voxel_size: [entity]
    spliced_dataset._get_segmentation_masks = lambda _root, _voxel_size: {"particle": entity}
    spliced_dataset._load_experimental_zarr()
    spliced_dataset._load_synthetic_zarr()
    spliced_dataset._load_segmentation_masks()

    with patch("copick.util.uri.resolve_copick_objects", return_value=[entity]):
        nnunet_volume = load_volume(root, "copick://tomogram", "run-1")
        nnunet_segmentation = load_segmentation(root, "copick://segmentation", "run-1")

    minimal_patch, minimal_label = minimal_dataset[0]
    return {
        "copick-patches": copick_dataset._subvolumes,
        "copick-labels": copick_dataset._molecule_ids,
        "simple-patches": simple_dataset._subvolumes,
        "simple-labels": simple_dataset._molecule_ids,
        "minimal-patch": minimal_patch.numpy(),
        "minimal-label": minimal_label,
        "spliced-experimental": spliced_dataset._exp_zarr_data,
        "spliced-synthetic": spliced_dataset._synth_zarr_data,
        "spliced-mask": spliced_dataset._synth_mask_data["particle"],
        "nnunet-volume": nnunet_volume,
        "nnunet-segmentation": nnunet_segmentation,
    }


@pytest.mark.parametrize("layout", V3_LAYOUTS)
def test_dataset_consumers_are_equivalent_across_v2_and_v3_layouts(tmp_path, layout):
    legacy_path = tmp_path / "legacy.zarr"
    migrated_path = tmp_path / f"{layout}.zarr"
    legacy, expected = write_store(legacy_path, "legacy-v2-numeric")
    migrated, _ = write_store(migrated_path, layout)
    before = snapshot(migrated_path)

    legacy_result = consume(legacy)
    migrated_result = consume(migrated)

    assert set(legacy_result) == set(migrated_result)
    for key in legacy_result:
        np.testing.assert_array_equal(migrated_result[key], legacy_result[key])
    np.testing.assert_array_equal(migrated.numpy(), expected)
    assert snapshot(migrated_path) == before
