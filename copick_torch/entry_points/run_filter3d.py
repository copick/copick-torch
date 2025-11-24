import click


def low_pass_commands(func):
    """Decorator to add common options to a Click command."""
    options = [
        click.option(
            "--lp-freq",
            type=float,
            required=False,
            default=0,
            help="Low-pass cutoff frequency (in Angstroms)",
        ),
        click.option("--lp-decay", type=float, required=False, default=50, help="Low-pass decay width (in pixels)"),
        click.option(
            "--hp-freq",
            type=float,
            required=False,
            default=0,
            help="High-pass cutoff frequency (in Angstroms)",
        ),
        click.option("--hp-decay", type=float, required=False, default=50, help="High-pass decay width (in pixels)"),
        click.option(
            "--show-filter",
            type=bool,
            required=False,
            default=True,
            help="Save the filter as a PNG (filter3d.png)",
        ),
    ]
    for option in reversed(options):  # Add options in reverse order to preserve correct order
        func = option(func)
    return func


def copick_commands(func):
    """Decorator to add common options to a Click command."""
    options = [
        click.option("--config", type=str, required=True, help="Path to Copick Config for Processing Data"),
        click.option(
            "--run-ids",
            type=str,
            required=False,
            default=None,
            help="Run ID to process (No Input would process the entire dataset.)",
        ),
        click.option("--tomo-alg", type=str, required=True, help="Tomogram Algorithm to use"),
        click.option("--voxel-size", type=float, required=False, default=10, help="Voxel Size to Query the Data"),
    ]
    for option in reversed(options):  # Add options in reverse order to preserve correct order
        func = option(func)
    return func


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


@click.command(context_settings={"show_default": True})
@copick_commands
@low_pass_commands
def bandpass(
    config: str,
    run_ids: str,
    lp_freq: float,
    lp_decay: float,
    hp_freq: float,
    hp_decay: float,
    tomo_alg: str,
    voxel_size: float,
    show_filter: bool,
):
    """
    3D bandpass filter tomograms.
    """

    run_filter3d(config, run_ids, lp_freq, lp_decay, hp_freq, hp_decay, tomo_alg, voxel_size, show_filter)


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
):
    import os

    import copick

    from copick_torch import parallelization
    from copick_torch.filters.bandpass import init_filter3d, run_filter3d

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

    # Get Run IDs
    if run_ids is None:
        run_ids = [run.name for run in root.runs]
    else:
        run_ids = run_ids.split(",")

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
    print("✅ Completed the Filtering!")


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
        target = np.zeros(shape, dtype=np.uint8)

        return target.shape


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
