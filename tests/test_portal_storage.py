import os
from pathlib import Path

import numpy as np
import pytest

from copick_torch.storage import get_level_array

MAX_PORTAL_VOXELS = 4 * 4 * 4
PORTAL_CONFIG = Path(__file__).parents[1] / "examples" / "czii_object_detection_training.json"


@pytest.mark.portal
@pytest.mark.skipif(os.environ.get("COPICK_TORCH_RUN_PORTAL") != "1", reason="bounded portal smoke not requested")
def test_public_portal_metadata_and_bounded_slice():
    import copick

    root = copick.from_file(str(PORTAL_CONFIG))
    tomograms = [tomogram for run in root.runs for spacing in run.voxel_spacings for tomogram in spacing.tomograms]
    assert tomograms, "portal configuration resolved no tomograms"

    array = get_level_array(tomograms[0])
    selection = tuple(slice(0, min(4, size)) for size in array.shape)
    requested_voxels = int(np.prod([part.stop - part.start for part in selection]))
    assert 0 < requested_voxels <= MAX_PORTAL_VOXELS

    decoded = np.asarray(array[selection])
    assert decoded.shape == tuple(part.stop for part in selection)
    assert decoded.size == requested_voxels
