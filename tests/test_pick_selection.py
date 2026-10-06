"""Particle centres include the pick's shift, and filament objects are not particle classes."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from copick_torch.copick import CopickDataset
from copick_torch.dataset import SimpleCopickDataset
from copick_torch.minimal_dataset import MinimalCopickDataset
from copick_torch.pick_utils import is_filament, pick_centres, selection_cache_suffix

VOXEL = 10.0
TOMOGRAM = np.zeros((64, 64, 64), dtype=np.float32)


def _transforms(shifts):
    transforms = np.tile(np.eye(4), (len(shifts), 1, 1))
    transforms[:, :3, 3] = shifts
    return transforms


def _picks(name, locations, shifts):
    picks = MagicMock()
    picks.from_tool = True
    picks.pickable_object_name = name
    picks.numpy.return_value = (np.asarray(locations, dtype=float), _transforms(shifts))
    return picks


def _root():
    """One particle object with a shifted pick, one filament object with two picks along a line."""
    ribosome = SimpleNamespace(name="ribosome", label=1, metadata={})
    microtubule = SimpleNamespace(name="microtubule", label=2, metadata={"copick": {"filament": {"polar": True}}})
    tomogram = MagicMock()
    tomogram.numpy.return_value = TOMOGRAM
    tomogram.tomo_type = "wbp"
    voxel_spacing = MagicMock()
    voxel_spacing.tomograms = [tomogram]
    run = MagicMock()
    run.name = "TS-1"
    run.get_voxel_spacing.return_value = voxel_spacing
    run.get_picks.return_value = [
        # location (160, 160, 160) + t (40, 0, 0) A = voxel (20, 16, 16)
        _picks("ribosome", [[160.0, 160.0, 160.0]], [[40.0, 0.0, 0.0]]),
        _picks("microtubule", [[300.0, 300.0, 300.0], [400.0, 300.0, 300.0]], [[0.0, 0.0, 0.0]] * 2),
    ]
    root = MagicMock()
    root.runs = [run]
    root.pickable_objects = [ribosome, microtubule]
    return root


def test_pick_centres_add_the_translation():
    picks = _picks("ribosome", [[1.0, 2.0, 3.0]], [[10.0, 20.0, 30.0]])
    assert np.allclose(pick_centres(picks), [[11.0, 22.0, 33.0]])
    picks.numpy.return_value = (np.array([[1.0, 2.0, 3.0]]), None)
    assert np.allclose(pick_centres(picks), [[1.0, 2.0, 3.0]])


def test_is_filament_reads_the_property_or_the_metadata():
    assert is_filament(SimpleNamespace(is_filament=True))
    assert is_filament(SimpleNamespace(metadata={"copick": {"filament": {}}}))
    assert not is_filament(SimpleNamespace(metadata={"copick": {"filament": None}}))
    assert not is_filament(MagicMock())


@pytest.mark.parametrize("dataset_class", [CopickDataset, SimpleCopickDataset])
@pytest.mark.parametrize(
    ("options", "classes", "extracted"),
    [
        ({}, ["ribosome"], [(20, 16, 16)]),
        ({"include_filaments": True}, ["ribosome", "microtubule"], [(20, 16, 16), (30, 30, 30), (40, 30, 30)]),
        ({"object_names": ["microtubule"], "include_filaments": True}, ["microtubule"], [(30, 30, 30), (40, 30, 30)]),
    ],
)
def test_samples_are_particle_centres_of_the_selected_objects(dataset_class, options, classes, extracted):
    seen = []

    def extract(self, tomogram_array, x, y, z):
        seen.append((x, y, z))
        return np.zeros((8, 8, 8), dtype=np.float32), True, "valid"

    with patch.object(dataset_class, "_extract_subvolume_with_validation", extract):
        dataset = dataset_class(copick_root=_root(), boxsize=(8, 8, 8), voxel_spacing=VOXEL, cache_dir=None, **options)
    assert dataset._keys == classes
    assert np.allclose(seen, extracted)


@pytest.mark.parametrize("dataset_class", [CopickDataset, SimpleCopickDataset])
def test_background_keeps_away_from_skipped_filaments(dataset_class):
    with (
        patch.object(dataset_class, "_extract_subvolume_with_validation") as extract,
        patch.object(
            dataset_class,
            "_sample_background_points",
        ) as sample_background,
    ):
        extract.return_value = (np.zeros((8, 8, 8), dtype=np.float32), True, "valid")
        dataset_class(
            copick_root=_root(),
            boxsize=(8, 8, 8),
            voxel_spacing=VOXEL,
            cache_dir=None,
            include_background=True,
        )
    _, particles, excluded = sample_background.call_args.args
    assert np.allclose(particles, [(20, 16, 16)])
    assert np.allclose(excluded, [(30, 30, 30), (40, 30, 30)])


def test_minimal_dataset_skips_filaments_and_uses_centres():
    with patch("copick_torch.minimal_dataset.get_level_array", return_value=TOMOGRAM):
        dataset = MinimalCopickDataset(proj=_root(), boxsize=(8, 8, 8), voxel_spacing=VOXEL, preload=False)
        with_filaments = MinimalCopickDataset(
            proj=_root(),
            boxsize=(8, 8, 8),
            voxel_spacing=VOXEL,
            preload=False,
            include_filaments=True,
        )
    assert dataset._name_to_label == {"ribosome": 1}
    assert np.allclose(dataset._points, [(200.0, 160.0, 160.0)])
    assert with_filaments._name_to_label == {"ribosome": 1, "microtubule": 2}
    assert len(with_filaments._points) == 3


def test_selection_is_part_of_the_cache_key():
    suffixes = {
        selection_cache_suffix(None, False),
        selection_cache_suffix(None, True),
        selection_cache_suffix(["ribosome"], False),
        selection_cache_suffix(["microtubule", "ribosome"], False),
    }
    assert len(suffixes) == 4
    assert selection_cache_suffix(["b", "a"], False) == selection_cache_suffix(["a", "b"], False)
