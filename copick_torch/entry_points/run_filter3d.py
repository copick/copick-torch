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
        click.option("--lp-decay", type=float, required=False, default=0, help="Low-pass decay width (in pixels)"),
        click.option(
            "--hp-freq",
            type=float,
            required=False,
            default=0,
            help="High-pass cutoff frequency (in Angstroms)",
        ),
        click.option("--hp-decay", type=float, required=False, default=0, help="High-pass decay width (in pixels)"),
        click.option(
            "--show-filter",
            type=bool,
            required=False,
            default=False,
            help="Save the filter as a Png (filter3d.png)",
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
        click.option(
            "--show-filter",
            type=bool,
            required=False,
            default=True,
            help="Save the filter as a Png (filter3d.png)",
        ),
    ]
    for option in reversed(options):  # Add options in reverse order to preserve correct order
        func = option(func)
    return func


def input_check(lp_freq, hp_freq, voxel_size):
    if lp_freq == 0 and hp_freq == 0:
        raise ValueError("Low-pass and high-pass frequencies cannot both be 0.")
    elif lp_freq < voxel_size * 2:
        raise ValueError("Low-pass frequency cannot be less than twice the Nyquist resolution.")
    elif hp_freq < voxel_size * 2:
        raise ValueError("High-pass frequency cannot be less than twice the Nyquist resolution.")
    elif lp_freq > hp_freq and lp_freq > 0 and hp_freq > 0:
        raise ValueError("Low-pass cutoff resolution must be less than high-pass cutoff resolution.")


def print_header(lp_freq, lp_decay, hp_freq, hp_decay):
    print("----------------------------------------")
    print(f"Low-Pass Frequency: {lp_freq} Angstroms")
    print(f"Low-Pass Decay: {lp_decay} Pixels")
    print(f"High-Pass Frequency: {hp_freq} Angstroms")
    print(f"High-Pass Decay: {hp_decay} Pixels")
    print("----------------------------------------")


@click.command(context_settings={"show_default": True})
@low_pass_commands
@copick_commands
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

    input_check(lp_freq, hp_freq, voxel_size)

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

    # Initialize Parallelization Pool
    pool = parallelization.GPUPool(
        init_fn=init_filter3d,
        init_args=(voxel_size, lp_freq, lp_decay, hp_freq, hp_decay),
        verbose=True,
    )

    # Execute
    tasks = [(run, tomo_alg, voxel_size, write_algorithm) for run in run_ids]
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
