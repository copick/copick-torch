import json
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from copick_torch.minimal_dataset import MinimalCopickDataset
from tests.storage_helpers import make_entity, make_v2_store


class TestMinimalCopickDataset(unittest.TestCase):
    """Exercise the eager and saved-lazy entity-store consumers."""

    def test_dataset_initialization_reads_numeric_ome_level(self):
        store, expected = make_v2_store("0")
        tomogram = make_entity(store)
        voxel_spacing = SimpleNamespace(tomograms=[tomogram])
        run = SimpleNamespace(
            name="run-1",
            get_voxel_spacing=lambda _voxel_size: voxel_spacing,
            get_picks=lambda: [],
        )
        project = SimpleNamespace(
            runs=[run],
            pickable_objects=[SimpleNamespace(name="particle", label=1)],
        )

        dataset = MinimalCopickDataset(proj=project, voxel_spacing=10.0, preload=False)

        self.assertEqual(len(dataset._tomogram_data), 1)
        np.testing.assert_array_equal(dataset._tomogram_data[0], expected)

    def test_lazy_reload_retains_numeric_zarr_array(self):
        store, expected = make_v2_store("0")
        tomogram = make_entity(store)
        tomogram.zarr = MagicMock(return_value=store)
        voxel_spacing = SimpleNamespace(tomograms=[tomogram])
        project = SimpleNamespace(
            runs=[SimpleNamespace(get_voxel_spacing=lambda _voxel_size: voxel_spacing)],
        )

        with TemporaryDirectory() as save_dir:
            metadata = {
                "dataset_id": None,
                "boxsize": [4, 4, 4],
                "voxel_spacing": 10.0,
                "include_background": False,
                "background_ratio": 0.0,
                "min_background_distance": 4,
                "name_to_label": {"particle": 1},
                "preload": False,
            }
            samples = [{"point": [20, 20, 20], "label": 1, "is_background": False, "tomogram_idx": 0}]
            tomograms = [{"index": index, "shape": list(expected.shape), "path": "0"} for index in range(3)]
            for name, value in (
                ("metadata.json", metadata),
                ("samples.json", samples),
                ("tomogram_info.json", tomograms),
            ):
                with open(f"{save_dir}/{name}", "w") as stream:
                    json.dump(value, stream)

            dataset = MinimalCopickDataset.load(save_dir, proj=project)

        self.assertEqual(dataset._tomogram_data[0].shape, expected.shape)
        np.testing.assert_array_equal(dataset._tomogram_data[0][1:3, 2:4, 3:5], expected[1:3, 2:4, 3:5])
        tomogram.zarr.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
