from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import zarr

from copick_torch.entry_points.run_filter3d import get_tomo_shape
from copick_torch.fitting.slab_from_picks import slab_from_picks
from tests.storage_helpers import make_entity, make_v3_store


class PayloadGuardStore(zarr.storage.MemoryStore):
    """Fail if a shape-only consumer asks the store for an array chunk."""

    def __init__(self, store_dict=None, *, read_only=False, payload_reads=None):
        super().__init__(store_dict=store_dict, read_only=read_only)
        self.payload_reads = [] if payload_reads is None else payload_reads

    def with_read_only(self, read_only=False):
        return type(self)(
            store_dict=self._store_dict,
            read_only=read_only,
            payload_reads=self.payload_reads,
        )

    async def get(self, key, prototype=None, byte_range=None):
        if key.startswith("s0/c/"):
            self.payload_reads.append(key)
            raise AssertionError(f"shape-only consumer attempted to read payload key {key!r}")
        return await super().get(key, prototype=prototype, byte_range=byte_range)


def shape_only_entity():
    store = PayloadGuardStore()
    make_v3_store("s0", shape=(128, 128, 128), chunks=(32, 32, 32), store=store)
    store.payload_reads.clear()
    return make_entity(store), store


def test_filter_shape_lookup_does_not_read_array_payload():
    tomogram, store = shape_only_entity()
    voxel_spacing = SimpleNamespace(get_tomogram=lambda _tomo_type: tomogram)
    run = SimpleNamespace(get_voxel_spacing=lambda _voxel_size: voxel_spacing)
    root = SimpleNamespace(get_run=lambda _run_name: run)

    assert get_tomo_shape(root, ["run-1"], "wbp", 10.0) == (128, 128, 128)
    assert store.payload_reads == []


def test_slab_shape_lookup_does_not_read_array_payload():
    tomogram, store = shape_only_entity()
    voxel_spacing = SimpleNamespace(get_tomogram=lambda _tomo_type: tomogram)
    run = SimpleNamespace(get_voxel_spacing=lambda _voxel_size: voxel_spacing)
    points = [[10.0, 10.0, 10.0], [20.0, 20.0, 10.0], [10.0, 20.0, 10.0]]
    picks = SimpleNamespace(
        points=[SimpleNamespace(location=SimpleNamespace(x=x, y=y, z=z)) for x, y, z in points],
    )
    surface = np.zeros((4, 3), dtype=np.float32)
    sentinel = object()

    with (
        patch(
            "copick_torch.fitting.slab_from_picks.fit_parallel_planes_from_picks",
            return_value=(np.array([0.0, 0.0, 1.0]), 0.25, 0.5),
        ),
        patch("copick_torch.fitting.slab_from_picks.evaluate_plane_on_grid", return_value=surface),
        patch("copick_torch.fitting.slab_from_picks.triangulate_box", return_value=sentinel),
        patch("copick_torch.fitting.slab_from_picks.store_mesh_with_stats", return_value=sentinel),
    ):
        result = slab_from_picks(
            picks,
            picks,
            run,
            "slab",
            "session",
            "user",
            "wbp",
            10.0,
            method="parallel",
        )

    assert result is sentinel
    assert store.payload_reads == []
