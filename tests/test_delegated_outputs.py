import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import zarr
from copick.impl.filesystem import CopickConfigFSSpec, CopickRootFSSpec
from copick.util.ome import get_level_path
from copick_utils.io import writers

from copick_torch.entry_points.run_membrane_seg import run_segmenter
from copick_torch.filters.bandpass import run_filter3d
from copick_torch.filters.downsample import run_downsampler
from copick_torch.nnunet.predict import single_gpu_nnUNetPredictor
from copick_torch.parallelization import GPUPool


def filesystem_run(tmp_path):
    root = CopickRootFSSpec(
        CopickConfigFSSpec(overlay_root=f"local://{tmp_path}", pickable_objects=[]),
    )
    return root.new_run("run-1")


def assert_canonical_output(entity, expected, codec_names):
    group = zarr.open_group(entity.zarr(), mode="r")
    array = group[get_level_path(group, 0)]

    assert group.metadata.zarr_format == 3
    assert group.attrs["ome"]["version"] == "0.5"
    assert array.metadata.dimension_names == ("z", "y", "x")
    assert array.chunks == (128, 128, 128)
    assert array.shards == (128, 128, 128)
    assert array.metadata.chunk_key_encoding.to_dict() == {"name": "v2", "configuration": {"separator": "/"}}
    sharding = array.metadata.to_dict()["codecs"][0]
    assert sharding["name"] == "sharding_indexed"
    assert [codec["name"] for codec in sharding["configuration"]["codecs"]] == codec_names
    np.testing.assert_array_equal(array[:], expected)


def test_delegated_utility_writers_emit_canonical_core_outputs(tmp_path):
    run = filesystem_run(tmp_path)
    tomogram = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    segmentation = (tomogram % 4).astype(np.uint16)

    writers.tomogram(run, tomogram, voxel_size=10.0, algorithm="filtered")
    writers.segmentation(
        run,
        segmentation,
        user_id="tester",
        name="membranes",
        session_id="1",
        voxel_size=10.0,
    )

    tomo_entity = run.get_voxel_spacing(10.0).get_tomogram("filtered")
    seg_entity = run.get_segmentations(name="membranes", user_id="tester", session_id="1")[0]
    assert_canonical_output(tomo_entity, tomogram, ["bytes", "numcodecs.shuffle", "zstd"])
    assert_canonical_output(seg_entity, segmentation.astype(np.uint8), ["bytes", "zstd"])
    np.testing.assert_array_equal(tomo_entity.numpy(), tomogram)
    np.testing.assert_array_equal(seg_entity.numpy(), segmentation.astype(np.uint8))


def test_downsample_keeps_reader_and_writer_delegation():
    run = SimpleNamespace(name="run-1")
    source = np.ones((4, 4, 4), dtype=np.float32)
    output = np.full((2, 2, 2), 2.0, dtype=np.float32)
    model = SimpleNamespace(run=MagicMock(return_value=output))

    with (
        patch("copick_utils.io.readers.tomogram", return_value=source) as reader,
        patch("copick_utils.io.writers.tomogram") as writer,
    ):
        run_downsampler(run, "wbp", 10.0, 20.0, False, "downsampled", gpu_id=0, models=model)

    reader.assert_called_once_with(run, 10.0, "wbp")
    model.run.assert_called_once_with(source)
    writer.assert_called_once_with(run, output, 20.0, "downsampled")


def test_bandpass_keeps_reader_and_writer_delegation():
    run = SimpleNamespace(name="run-1")
    source = np.ones((4, 4, 4), dtype=np.float32)
    output = torch.full((4, 4, 4), 3.0)
    model = SimpleNamespace(apply=MagicMock(return_value=output))

    with (
        patch("copick_utils.io.readers.tomogram", return_value=source) as reader,
        patch("copick_utils.io.writers.tomogram") as writer,
    ):
        run_filter3d(run, "wbp", 10.0, "bandpass", gpu_id=0, models=model)

    reader.assert_called_once_with(run, 10.0, "wbp")
    model.apply.assert_called_once_with(source)
    np.testing.assert_array_equal(writer.call_args.args[1], output.numpy())
    assert writer.call_args.args[0] is run
    assert writer.call_args.args[2:] == (10.0, "bandpass")


def test_membrain_keeps_reader_and_writer_delegation():
    run = SimpleNamespace(name="run-1")
    source = np.ones((4, 4, 4), dtype=np.float32)
    output = np.ones((4, 4, 4), dtype=np.uint8)

    with (
        patch("copick_utils.io.readers.tomogram", return_value=source) as reader,
        patch("copick_torch.inference.membrain_seg.membrain_segment", return_value=output) as segment,
        patch("copick_utils.io.writers.segmentation") as writer,
    ):
        run_segmenter(run, "wbp", 10.0, "session", 0.5, "tester", gpu_id=0, models=object())

    reader.assert_called_once_with(run, 10.0, "wbp")
    assert segment.call_args.args[0] is source
    writer.assert_called_once_with(
        run,
        output,
        "tester",
        "membranes",
        session_id="session",
        voxel_size=10.0,
    )


def test_nnunet_keeps_entity_read_and_writer_delegation():
    source = np.ones((4, 4, 4), dtype=np.float32)
    output = np.ones((4, 4, 4), dtype=np.uint8)
    volume = SimpleNamespace(numpy=MagicMock(return_value=source))
    run = SimpleNamespace(name="run-1")
    root = SimpleNamespace(runs=[run])
    predictor = single_gpu_nnUNetPredictor.__new__(single_gpu_nnUNetPredictor)
    predictor.predict = MagicMock(return_value=output)

    with (
        patch("copick.from_file", return_value=root),
        patch("copick.util.uri.resolve_copick_objects", return_value=[volume]),
        patch("copick_utils.io.writers.segmentation") as writer,
    ):
        predictor.batch_predict("config.json", "wbp@10.0", "prediction:tester/1")

    volume.numpy.assert_called_once_with()
    predictor.predict.assert_called_once_with(source, voxel_size_angstrom=10.0)
    writer.assert_called_once_with(
        run,
        output,
        voxel_size=10.0,
        name="prediction",
        user_id="tester",
        session_id="1",
    )


def test_core_writer_rejection_survives_gpu_pool_result(tmp_path, monkeypatch):
    from copick.util import ome

    run = filesystem_run(tmp_path)
    monkeypatch.setattr(ome, "_SHARD_LIMIT_BYTES", 1)
    pool = GPUPool.__new__(GPUPool)
    pool.n_gpus = 1
    pool.verbose = False
    pool.initialized = threading.Event()
    pool.initialized.set()
    pool.models = {0: None}
    pool.model_locks = {0: threading.RLock()}

    def rejected_write(gpu_id):
        assert gpu_id == 0
        writers.tomogram(run, np.ones((2, 2, 2), dtype=np.float32), 10.0, "too-large")

    with patch("torch.cuda.set_device"):
        results = pool._execute_threading(rejected_write, [()], ["run-1"], "test")

    assert results[0]["success"] is False
    assert "below 5 TB" in results[0]["error"]
