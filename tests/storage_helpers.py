from types import SimpleNamespace

import numpy as np
import zarr


class CountingMemoryStore(zarr.storage.MemoryStore):
    """In-memory Zarr v2 store that records reads of array payload keys."""

    def __init__(self):
        super().__init__()
        self.payload_reads = []

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if not key.endswith((".zarray", ".zattrs", ".zgroup")):
            self.payload_reads.append(key)
        return value


def make_v2_store(dataset_path="0", data=None):
    """Build a small metadata-valid OME-Zarr 0.4 / Zarr v2 group."""
    if data is None:
        data = np.arange(8 * 9 * 10, dtype=np.float32).reshape(8, 9, 10)

    store = CountingMemoryStore()
    group = zarr.group(store=store, overwrite=True)
    group.create_dataset(dataset_path, data=data, chunks=(4, 4, 4))
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
    store.payload_reads.clear()
    return store, np.asarray(data)


def make_entity(store, *, tomo_type="wbp-denoised", name="particle"):
    """Return the smallest entity double needed by the storage consumers."""
    return SimpleNamespace(
        tomo_type=tomo_type,
        meta=SimpleNamespace(name=name),
        zarr=lambda: store,
    )
