"""Tests for the Part 2 CLI flag normalization in copick-torch.

downsample and bandpass now take tomogram URIs via the shared `-i`/`-o` options;
their old `--tomo-alg`/`--voxel-size`/`--target-resolution` params remain as
hidden deprecated fallbacks.
"""

import pytest
from click.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


def _load(dotted, attr):
    import importlib

    return getattr(importlib.import_module(dotted), attr)


def _hidden(cmd, flag):
    param = next((p for p in cmd.params if flag in getattr(p, "opts", [])), None)
    return param is not None and param.hidden


def test_downsample_uses_io_uris(runner):
    cmd = _load("copick_torch.entry_points.run_downsample", "downsample")
    out = runner.invoke(cmd, ["--help"]).output
    assert "-i" in out and "--input" in out
    assert "-o" in out and "--output" in out
    for legacy in ("--tomo-alg", "--voxel-size", "--target-resolution"):
        assert legacy not in out
        assert _hidden(cmd, legacy)


def test_bandpass_uses_io_uris(runner):
    cmd = _load("copick_torch.entry_points.run_filter3d", "bandpass")
    out = runner.invoke(cmd, ["--help"]).output
    assert "-i" in out and "--input" in out
    assert "-o" in out and "--output" in out
    for legacy in ("--tomo-alg", "--voxel-size"):
        assert legacy not in out
        assert _hidden(cmd, legacy)
