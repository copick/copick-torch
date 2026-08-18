from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from copick_torch.entry_points.run_filter3d import get_tomo_shape
from copick_torch.fitting.slab_from_picks import slab_from_picks
from tests.storage_helpers import make_entity, make_v2_store


class ShapeOnlyArray:
    shape = (8, 9, 10)

    def __getitem__(self, _selection):
        raise AssertionError("shape-only consumer attempted to read array data")

    def __array__(self, *_args, **_kwargs):
        raise AssertionError("shape-only consumer attempted to materialize the array")


def test_filter_shape_lookup_does_not_read_array_payload():
    tomogram = make_entity(object())
    voxel_spacing = SimpleNamespace(get_tomogram=lambda _tomo_type: tomogram)
    run = SimpleNamespace(get_voxel_spacing=lambda _voxel_size: voxel_spacing)
    root = SimpleNamespace(get_run=lambda _run_name: run)

    with patch("copick_torch.storage.get_level_array", return_value=ShapeOnlyArray()):
        assert get_tomo_shape(root, ["run-1"], "wbp", 10.0) == ShapeOnlyArray.shape


def test_slab_shape_lookup_does_not_read_array_payload():
    tomogram = make_entity(object())
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
        patch("copick_torch.storage.get_level_array", return_value=ShapeOnlyArray()),
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


def test_numeric_and_nonnumeric_v2_fixtures_are_decoded_equivalent():
    numeric_store, expected = make_v2_store("0")
    nonnumeric_store, _ = make_v2_store("s0", expected)

    numeric = np.asarray(__import__("zarr").open(numeric_store, mode="r")["0"])
    nonnumeric = np.asarray(__import__("zarr").open(nonnumeric_store, mode="r")["s0"])

    np.testing.assert_array_equal(numeric, nonnumeric)
