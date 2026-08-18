"""Read-only access to metadata-defined arrays in copick entity stores."""

from typing import Any

import zarr
from copick.util.ome import get_level_path


def get_level_array(entity: Any, level: int = 0) -> zarr.Array:
    """Return a lazy array for a zero-based OME multiscale level.

    The level is resolved from OME metadata and is never interpreted as a
    literal dataset name. Source layout details such as chunks, shards, keys,
    and codecs are deliberately left to Zarr.
    """
    store = entity.zarr()
    if store is None:
        raise ValueError(f"{type(entity).__name__}.zarr() returned no store")

    group = zarr.open_group(store=store, mode="r")
    return group[get_level_path(group, level)]
