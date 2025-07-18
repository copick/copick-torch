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
            run_downsampler,
            tasks,
            task_ids=run_ids,
            progress_desc="Downsampling Tomograms",
        )
    finally:
        pool.shutdown()

    print("Completed the Downsampling!")


def run_downsampler(run, tomo_alg, voxel_size, target_resolution, delete_source, gpu_id, models):
    from copick_utils.io import readers, writers

    # Get the Downsampler
    downsampler = models

    # Get the Tomogram
    tomo = readers.tomogram(run, voxel_size, tomo_alg)

    # Downsample the Tomogram
    downsampled_tomo = downsampler.run(tomo)

    # Save the Downsampled Tomogram
    writers.tomogram(run, downsampled_tomo, target_resolution, tomo_alg)

    # Delete the source tomograms if requested
    if delete_source:
        vs = run.get_voxel_spacing(voxel_size)
        vs.delete_tomograms(tomo_alg)

        # If the Voxel Spacing is Empty, lets delete it as well
        if vs.tomograms == []:
            vs.delete()
