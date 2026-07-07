"""CLI command for 3D band-pass filtering of tomograms on the GPU."""

import click
from click_option_group import optgroup
from copick.cli.util import (
    add_config_option,
    add_debug_option,
    add_deprecated_run_alias,
    add_run_names_option,
    resolve_run_names,
)
from copick.util.log import get_logger


def input_check(lp_res, hp_res, apix):
    """
    lp_res, hp_res: resolutions in Å (0 means 'disabled')
    apix: pixel size in Å/pixel
    require_filter: if True, disallow the all-pass case (both 0)
    """
    nyquist_res = 2.0 * apix  # smallest physically valid resolution (Å)

    # All-pass allowed unless explicitly forbidden
    if lp_res == 0 and hp_res == 0:
        raise ValueError("Low-pass and high-pass cannot both be 0 (no filtering).")

    # Low-pass: if enabled, it cannot be finer (smaller Å) than Nyquist
    if lp_res > 0 and lp_res < nyquist_res:
        raise ValueError(f"Low-pass resolution {lp_res:.3g} Å is finer than Nyquist (2*apix = {nyquist_res:.3g} Å).")

    # High-pass: if enabled, it also cannot be finer than Nyquist (no frequencies exist beyond Nyquist)
    if hp_res > 0 and hp_res < nyquist_res:
        raise ValueError(f"High-pass resolution {hp_res:.3g} Å is finer than Nyquist (2*apix = {nyquist_res:.3g} Å).")

    # Band-pass consistency: in Å, larger number = lower spatial frequency
    # Require hp > lp when both are enabled
    if lp_res > 0 and hp_res > 0 and not (hp_res > lp_res):
        raise ValueError("For band-pass, require hp (Å) > lp (Å).")


def print_header(lp_freq, lp_decay, hp_freq, hp_decay):
    print("----------------------------------------")
    print(f"Low-Pass Frequency: {lp_freq} Angstroms")
    print(f"Low-Pass Decay: {lp_decay} Pixels")
    print(f"High-Pass Frequency: {hp_freq} Angstroms")
    print(f"High-Pass Decay: {hp_decay} Pixels")
    print("----------------------------------------")


@click.command(
    "bandpass",
    context_settings={"show_default": True},
    short_help="Band-pass filter tomograms in 3D.",
    no_args_is_help=True,
)
@add_config_option
@add_run_names_option
@add_deprecated_run_alias("--run-ids")
@optgroup.group("\nInput Options", help="Options related to the input tomograms.")
@optgroup.option("--tomo-alg", type=str, required=True, help="Tomogram Algorithm to use")
@optgroup.option("--voxel-size", type=float, required=False, default=10, help="Voxel Size to Query the Data")
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--lp-freq",
    type=float,
    required=False,
    default=0,
    help="Low-pass cutoff frequency (in Angstroms)",
)
@optgroup.option("--lp-decay", type=float, required=False, default=50, help="Low-pass decay width (in pixels)")
@optgroup.option(
    "--hp-freq",
    type=float,
    required=False,
    default=0,
    help="High-pass cutoff frequency (in Angstroms)",
)
@optgroup.option("--hp-decay", type=float, required=False, default=50, help="High-pass decay width (in pixels)")
@optgroup.option(
    "--show-filter",
    type=bool,
    required=False,
    default=True,
    help="Save the filter as a PNG (filter3d.png)",
)
@add_debug_option
def bandpass(
    config: str,
    run_names,
    legacy_run_names,
    tomo_alg: str,
    voxel_size: float,
    lp_freq: float,
    lp_decay: float,
    hp_freq: float,
    hp_decay: float,
    show_filter: bool,
    debug: bool,
):
    """
    Band-pass filter tomograms in 3D.

    For every run in the project (or only the runs given by --run-names/-r), queries the
    tomogram of the given algorithm at the requested voxel spacing, applies a Gaussian
    band-pass filter on the GPU, and writes the filtered tomogram back into the project
    under a new algorithm name (the source name with -lp<freq>A and/or -hp<freq>A
    suffixes). Runs are processed in parallel on a GPU pool.

    Low-pass and high-pass cutoffs are given as resolutions in angstroms; a value of 0
    disables that side of the filter, but both cannot be 0. Cutoffs finer than Nyquist
    (2 * voxel_size) are rejected, and for a true band-pass the high-pass resolution must
    be coarser than the low-pass resolution. Optionally writes a PNG preview of the filter
    profile (filter3d.png).

    URI Format:

        \b
        Tomograms: tomo_type@voxel_spacing

    Examples:

        \b
        # Low-pass filter wbp tomograms at 10 A voxel spacing to 30 A resolution
        copick process bandpass -c config.json --tomo-alg wbp \\
            --voxel-size 10.0 --lp-freq 30.0

        \b
        # Band-pass a single run between 100 A (high-pass) and 30 A (low-pass)
        copick process bandpass -c config.json --tomo-alg wbp \\
            --voxel-size 10.0 -r TS_001 --lp-freq 30.0 --hp-freq 100.0

        \b
        # High-pass the whole dataset and skip the filter preview PNG
        copick process bandpass -c config.json --tomo-alg wbp \\
            --voxel-size 10.0 --hp-freq 150.0 --show-filter false

    See Also:

        \b
        copick process downsample: downsample tomograms via Fourier rescaling
    """

    logger = get_logger(__name__, debug=debug)
    run_names_list = resolve_run_names(run_names, legacy_run_names, legacy_flag="--run-ids", logger=logger)

    run_filter3d(
        config,
        run_names_list,
        lp_freq,
        lp_decay,
        hp_freq,
        hp_decay,
        tomo_alg,
        voxel_size,
        show_filter,
        debug=debug,
    )


def run_filter3d(
    config: str,
    run_ids: str,
    lp_freq: float,
    lp_decay: float,
    hp_freq: float,
    hp_decay: float,
    tomo_alg: str,
    voxel_size: float,
    show_filter: bool,
    debug: bool = False,
):
    import os

    import copick

    from copick_torch import parallelization
    from copick_torch.filters.bandpass import init_filter3d, run_filter3d

    logger = get_logger(__name__, debug=debug)

    # Input Check - Set Decay to 0 if Unused
    input_check(lp_freq, hp_freq, voxel_size)
    if lp_freq == 0:
        lp_decay = 0
    if hp_freq == 0:
        hp_decay = 0

    # Load Copick Project
    if os.path.exists(config):
        root = copick.from_file(config)
    else:
        raise ValueError(f"Config file {config} does not exist.")

    print_header(lp_freq, lp_decay, hp_freq, hp_decay)

    # Get Run IDs (already resolved to a list of run names, or None for all runs)
    if run_ids is None:
        run_ids = [run.name for run in root.runs]

    # Determine Write Algorithm
    write_algorithm = tomo_alg
    if lp_freq > 0:
        write_algorithm = write_algorithm + f"-lp{lp_freq:0.0f}A"
    if hp_freq > 0:
        write_algorithm = write_algorithm + f"-hp{hp_freq:0.0f}A"

    # Get Volume Shape
    vol_shape = get_tomo_shape(root, run_ids, tomo_alg, voxel_size)

    # Initialize Parallelization Pool
    pool = parallelization.GPUPool(
        init_fn=init_filter3d,
        init_args=(voxel_size, vol_shape, lp_freq, lp_decay, hp_freq, hp_decay),
        verbose=True,
    )
    # Save Filter Image
    if show_filter:
        save_filter((voxel_size, vol_shape, lp_freq, lp_decay, hp_freq, hp_decay))

    # Execute
    tasks = [(root.get_run(run), tomo_alg, voxel_size, write_algorithm) for run in run_ids]
    try:
        pool.execute(
            run_filter3d,
            tasks,
            task_ids=run_ids,
            progress_desc="Filtering Tomograms",
        )
    finally:
        pool.shutdown()

    save_parameters(config, [tomo_alg, voxel_size], [lp_freq, lp_decay, hp_freq, hp_decay], write_algorithm)
    logger.info("✅ Completed the Filtering!")


def get_tomo_shape(root, run_ids, tomo_alg, voxel_size):
    import numpy as np
    import zarr

    for runID in run_ids:
        # Get Volume Shape from First Run
        run = root.get_run(runID)

        # Get Target Shape
        vs = run.get_voxel_spacing(voxel_size)
        if vs is None:
            continue
        tomo = vs.get_tomogram(tomo_alg)
        if tomo is None:
            continue
        loc = tomo.zarr()
        shape = zarr.open(loc)["0"].shape
        return shape


def save_filter(params):
    import torch

    from copick_torch.filters.bandpass import Filter3D

    filter = Filter3D(
        params[0],
        params[1],
        params[2],
        params[3],
        params[4],
        params[5],
        device=torch.device("cpu"),
    )

    filter.show_filter()

    del filter


def save_parameters(config, tomo_info, parameters, write_algorithm):
    import os

    import copick

    from copick_torch.entry_points.utils import save_parameters_yaml

    root = copick.from_file(config)
    overlay_root = root.config.overlay_root
    if overlay_root[:8] == "local://":
        overlay_root = overlay_root[8:]
    group = {
        "input": {
            "config": config,
            "tomo_alg": tomo_info[0],
            "voxel_size": tomo_info[1],
        },
        "parameters": {
            "Low-Pass Frequency (Angstroms)": parameters[0],
            "Low-Pass Decay (Pixels)": parameters[1],
            "High-Pass Frequency (Angstroms)": parameters[2],
            "High-Pass Decay (Pixels)": parameters[3],
        },
        "output": {
            "tomo_alg": write_algorithm,
            "voxel_size": tomo_info[1],
        },
    }
    os.makedirs(os.path.join(overlay_root, "logs"), exist_ok=True)
    path = os.path.join(overlay_root, "logs", f"process-filter3d_{tomo_info[0]}_{tomo_info[1]}A.yaml")
    save_parameters_yaml(group, path)
    print(f"📝 Saved Parameters to {path}")
