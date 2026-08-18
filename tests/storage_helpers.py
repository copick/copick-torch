from types import SimpleNamespace

import numpy as np
import zarr


def make_v2_store(dataset_path="0", data=None):
    """Build a small metadata-valid OME-Zarr 0.4 / Zarr v2 group."""
    if data is None:
        data = np.arange(8 * 9 * 10, dtype=np.float32).reshape(8, 9, 10)

    store = zarr.storage.MemoryStore()
    group = zarr.open_group(store=store, mode="w", zarr_format=2)
    group.create_array(dataset_path, data=data, chunks=(4, 4, 4))
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
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
    ]
    return store, np.asarray(data)


def make_v3_store(dataset_path="s0", data=None, *, shape=None, store=None, chunks=(4, 4, 4)):
    """Build a metadata-valid OME-Zarr 0.5 / Zarr v3 group."""
    if data is None and shape is None:
        data = np.arange(8 * 9 * 10, dtype=np.float32).reshape(8, 9, 10)

    store = store or zarr.storage.MemoryStore()
    group = zarr.open_group(store=store, mode="w", zarr_format=3)
    if data is None:
        group.create_array(dataset_path, shape=shape, dtype=np.float32, chunks=chunks)
        expected = None
    else:
        group.create_array(dataset_path, data=data, chunks=chunks)
        expected = np.asarray(data)
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
    return store, expected


def make_entity(store, *, tomo_type="wbp-denoised", name="particle"):
    """Return the smallest entity double needed by the storage consumers."""
    return SimpleNamespace(
        tomo_type=tomo_type,
        meta=SimpleNamespace(name=name),
        zarr=lambda: store,
    )
