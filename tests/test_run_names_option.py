"""Tests that copick-torch CLI commands use the standardized run-selection option.

Verifies each command exposes ``--run-names``/``-r`` (from ``copick.cli.util``) and
that the commands whose flag was renamed keep a hidden, still-working ``--run-ids``
deprecated alias.
"""

import pytest
from click.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


def _load_command(dotted, attr):
    import importlib

    return getattr(importlib.import_module(dotted), attr)


ENTRY_POINT_COMMANDS = [
    ("copick_torch.entry_points.run_filter3d", "bandpass"),
    ("copick_torch.entry_points.run_downsample", "downsample"),
    ("copick_torch.entry_points.run_membrane_seg", "membrain_seg"),
    ("copick_torch.entry_points.run_picks2slab", "picks2slab"),
    ("copick_torch.entry_points.run_seg2slab", "seg2slab"),
    ("copick_torch.nnunet.predict", "cli"),
]


@pytest.mark.parametrize(("dotted", "attr"), ENTRY_POINT_COMMANDS)
def test_command_exposes_standard_run_option(runner, dotted, attr):
    cmd = _load_command(dotted, attr)
    out = runner.invoke(cmd, ["--help"]).output
    assert "--run-names" in out
    assert "-r" in out
    assert "--run-ids" not in out  # renamed; alias is hidden


@pytest.mark.parametrize(
    ("dotted", "attr"),
    [
        ("copick_torch.entry_points.run_filter3d", "bandpass"),
        ("copick_torch.nnunet.predict", "cli"),
    ],
)
def test_deprecated_run_ids_alias_registered_hidden(dotted, attr):
    cmd = _load_command(dotted, attr)
    alias = next((p for p in cmd.params if "--run-ids" in getattr(p, "opts", [])), None)
    assert alias is not None, "deprecated --run-ids alias should still be registered"
    assert alias.hidden is True
    assert alias.name == "legacy_run_names"
