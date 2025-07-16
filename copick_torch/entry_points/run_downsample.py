import click

@click.group(name="downsample")
@click.pass_context
def cli(ctx):
    """Downsample tomograms to a target resolution."""
    pass

@cli.command(context_settings={'show_default': True})
def downsample(
    config: str,
    tomo_alg: str,
    voxel_size: float,
    target_resolution: float,
    delete_source: bool
    ):
    from copick_torch.filters import downsample
    from copick_torch import parallelization
    import copick

    root = copick.from_file(config)
    run_ids = [run.name for run in root.runs]

    pool = parallelization.GPUPool(
        init_fn=downsample.downsample_init,
        init_args=(voxel_size, target_resolution),
        approach="threading",
        verbose=True
    )

    tasks = [
        (run, tomo_alg, voxel_size, target_resolution, delete_source)
        for run in root.runs
    ]
    
    # Execute 
    try:
        pool.execute(
            run_downsampler,
            tasks, task_ids=run_ids,
            progress_desc="Downsampling Tomograms"
        )     
    finally:
        pool.shutdown()

    print('Completed the Downsampling!')

def run_downsampler(run, tomo_alg, voxel_size, target_resolution, delete_source, gpu_id, models):
    from copick_utils.io import readers, writers

    # Get the Downsampler
    downsampler = models

    # Get the Tomogram
    tomo = readers.tomogram(run, tomo_alg, voxel_size)

    # Downsample the Tomogram
    downsampled_tomo = downsampler.run(tomo)

    # Save the Downsampled Tomogram
    writers.tomogram(run, downsampled_tomo, target_resolution)

    # Delete the source tomograms if requested
    if delete_source:
        vs = run.get_voxel_spacing(voxel_size)
        vs.delete_tomograms(tomo_alg)