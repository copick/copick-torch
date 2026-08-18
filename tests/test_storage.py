import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import zarr

from copick_torch.storage import get_level_array
from tests.storage_helpers import make_entity, make_v2_store


def make_v3_store(dataset_path="s0", data=None):
    if data is None:
        data = np.arange(8 * 9 * 10, dtype=np.float32).reshape(8, 9, 10)

    store = zarr.storage.MemoryStore()
    group = zarr.open_group(store=store, mode="w", zarr_format=3)
    group.create_array(dataset_path, data=data, chunks=(4, 4, 4))
    group.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [
                    {"name": "z", "type": "space", "unit": "angstrom"},
                    {"name": "y", "type": "space", "unit": "angstrom"},
                    {"name": "x", "type": "space", "unit": "angstrom"},
                ],
                "datasets": [
                    {
                        "path": dataset_path,
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [10.0, 10.0, 10.0]},
                        ],
                    },
                ],
            },
        ],
    }
    return store, np.asarray(data)


@pytest.mark.parametrize("dataset_path", ["0", "s0"])
def test_get_level_array_reads_ome_zarr_04_paths(dataset_path):
    store, expected = make_v2_store(dataset_path)

    array = get_level_array(make_entity(store))

    assert isinstance(array, zarr.Array)
    np.testing.assert_array_equal(array[:], expected)


def test_get_level_array_reads_nested_ome_zarr_05_metadata():
    store, expected = make_v3_store("scale-zero")

    array = get_level_array(make_entity(store))

    assert isinstance(array, zarr.Array)
    np.testing.assert_array_equal(array[:], expected)


def test_get_level_array_opens_store_exactly_read_only():
    store, _ = make_v3_store()
    entity = make_entity(store)

    with patch("copick_torch.storage.zarr.open_group", wraps=zarr.open_group) as open_group:
        get_level_array(entity)

    open_group.assert_called_once_with(store=store, mode="r")


def test_get_level_array_rejects_missing_store():
    with pytest.raises(ValueError, match="returned no store"):
        get_level_array(SimpleNamespace(zarr=lambda: None))


@pytest.mark.parametrize("level", [-1, 1])
def test_get_level_array_rejects_invalid_level(level):
    store, _ = make_v3_store()

    with pytest.raises(ValueError, match=f"Level {level} not found"):
        get_level_array(make_entity(store), level=level)


def test_empty_store_failure_does_not_create_metadata():
    store = zarr.storage.MemoryStore()
    assert asyncio.run(store.is_empty(""))

    with pytest.raises(Exception, match="not found|No group|does not exist"):
        get_level_array(make_entity(store))

    assert asyncio.run(store.is_empty(""))
