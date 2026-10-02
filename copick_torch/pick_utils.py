"""Pick centres and object selection shared by the datasets."""

import hashlib
from typing import Any, Iterable, Optional, Set

import numpy as np


def pick_centres(picks: Any) -> np.ndarray:
    """(N, 3) particle centres of a pick set in Angstrom, x, y, z.

    copick's particle centre is ``location + t``, with ``t`` the translation of the pick's transform (tomogram
    frame); ``numpy()`` returns the location alone. This is ``CopickPicks.full_positions()`` (copick >= 1.28),
    computed here so older copick gives the same centres.
    """
    positions, transforms = picks.numpy()
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    if transforms is None:
        return positions
    transforms = np.asarray(transforms, dtype=float).reshape(-1, 4, 4)
    if len(transforms) != len(positions):
        return positions
    return positions + transforms[:, :3, 3]


def is_filament(obj: Any) -> bool:
    """Whether a pickable object is declared a filament.

    copick >= 1.28 has ``is_filament``; older copick carries the spec in ``metadata["copick"]["filament"]``.
    """
    flag = getattr(obj, "is_filament", None)
    if isinstance(flag, bool):
        return flag
    metadata = getattr(obj, "metadata", None)
    namespace = metadata.get("copick") if isinstance(metadata, dict) else None
    return isinstance(namespace, dict) and namespace.get("filament") is not None


def filament_object_names(pickable_objects: Iterable[Any]) -> Set[str]:
    """Names of the filament objects among ``pickable_objects``."""
    try:
        objects = list(pickable_objects)
    except TypeError:
        return set()
    return {obj.name for obj in objects if is_filament(obj)}


def skip_reason(
    object_name: str,
    filament_names: Set[str],
    object_names: Optional[Iterable[str]],
    include_filaments: bool,
) -> Optional[str]:
    """Why picks of ``object_name`` are not training samples, or None when they are.

    Picks along a filament sample one continuous structure, not separate particles, so filament objects are left
    out unless asked for.
    """
    if object_names is not None and object_name not in set(object_names):
        return "not in object_names"
    if not include_filaments and object_name in filament_names:
        return "filament object (include_filaments=False)"
    return None


def selection_cache_suffix(object_names: Optional[Iterable[str]], include_filaments: bool) -> str:
    """Cache-file suffix for an object selection, so datasets built from different selections never share a cache."""
    if object_names is None:
        selected = "all"
    else:
        selected = hashlib.sha1(",".join(sorted(object_names)).encode()).hexdigest()[:8]
    return f"_objs-{selected}{'_fil' if include_filaments else ''}"
