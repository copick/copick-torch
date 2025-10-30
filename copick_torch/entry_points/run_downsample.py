import click


def downsample_commands(func):
    """Decorator to add common options to a Click command."""
    options = [
        click.option("--config", type=str, required=True, help="Path to Copick Config for Processing Data"),
        click.option("--tomo-alg", type=str, required=True, help="Tomogram Algorithm to use"),
        click.option("--voxel-size", type=float, required=False, default=10, help="Voxel Size to Query the Data"),
        click.option(
            "--target-resolution",
            type=float,
            required=False,
            default=10,
            help="Target Resolution to Downsample to",
        ),
        click.option(
            "--delete-source",
            type=bool,
            required=False,
            default=False,
            help="Delete the source tomograms after downsampling",
        ),
    ]
    for option in reversed(options):  # Add options in reverse order to preserve correct order
        func = option(func)
    return func


@click.command(context_settings={"show_default": True})
@downsample_commands
@click.pass_context
def downsample(
    ctx: click.Context,
    config: str,
    tomo_alg: str,
    voxel_size: float,
    target_resolution: float,
    delete_source: bool,
):
    """
    Runs the downsampling command.
    """

    run(config, tomo_alg, voxel_size, target_resolution, delete_source)


def run(config, tomo_alg, voxel_size, target_resolution, delete_source):
    """
    Runs the downsampling.
    """

    import copick

    from copick_torch import parallelization
    from copick_torch.filters import downsample

    root = copick.from_file(config)
    run_ids = [run.name for run in root.runs]

    pool = parallelization.GPUPool(
        init_fn=downsample.downsample_init,
        init_args=(voxel_size, target_resolution),
        verbose=True,
    )

    tasks = [(run, tomo_alg, voxel_size, target_resolution, delete_source) for run in root.runs]

    # Execute
    try:
        pool.execute(
            downsample.run_downsampler,
            tasks,
            task_ids=run_ids,
            progress_desc="Downsampling Tomograms",
        )
    finally:
        pool.shutdown()

    save_parameters(config, tomo_alg, voxel_size, target_resolution)
    print("✅ Completed the Downsampling!")


def save_parameters(config, tomo_alg, voxel_size, target_resolution):
    """
    Save the parameters for the downsampling.
    """

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
            "tomo_alg": tomo_alg,
            "voxel_size": voxel_size,
        },
        "output": {
            "target_resolution": target_resolution,
        },
    }
    os.makedirs(os.path.join(overlay_root, "logs"), exist_ok=True)
    path = os.path.join(overlay_root, "logs", f"process-downsample_{tomo_alg}_{target_resolution}A.yaml")
    save_parameters_yaml(group, path)
    print(f"📝 Saved Parameters to {path}")
