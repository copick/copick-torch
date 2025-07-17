import click


def segment_commands(func):
    """Decorator to add common options to a Click command."""
    options = [
        click.option("--config", type=str, required=True, help="Path to Copick Config for Processing Data"),
        click.option("--tomo-alg", type=str, required=True, help="Tomogram Algorithm to use"),
        click.option("--voxel-size", type=float, required=False, default=10, help="Voxel Size to Query the Data"),
        click.option(
            "--session-id",
            type=str,
            required=False,
            default="1",
            help="Session ID for the Saved Membrane Segmentation",
        ),
        click.option(
            "--threshold",
            type=float,
            required=False,
            default=0,
            help="Segmentation Threshold for Membrane Segmentation",
        ),
    ]
    for option in reversed(options):  # Add options in reverse order to preserve correct order
        func = option(func)
    return func


@click.command(context_settings={"show_default": True})
@segment_commands
@click.pass_context
def membrain_seg(
    ctx: click.Context,
    config: str,
    tomo_alg: str,
    voxel_size: float,
    session_id: str,
    threshold: float,
):
    import copick

    from copick_torch import parallelization
    from copick_torch.inference import membrain_seg

    print("Starting Membrane Segmentation...")
    print(f"Using Tomograms with Voxel Size: {voxel_size} and Algorithm: {tomo_alg}")
    print(f"Saving Segmentations with membrain-seg_{session_id}_membranes Query")
    print(f"Segmentation Threshold: {threshold}\n")

    # Read Copick Project and Get Runs
    root = copick.from_file(config)
    run_ids = [run.name for run in root.runs]

    # Initialize Parallelization Pool
    pool = parallelization.GPUPool(
        init_fn=membrain_seg.membrane_seg_init,
        verbose=True,
    )

    # Check to see if model is available
    # If not, download it

    tasks = [(run, tomo_alg, voxel_size, session_id, threshold) for run in root.runs]

    # Execute
    try:
        pool.execute(
            run_segmenter,
            tasks,
            task_ids=run_ids,
            progress_desc="Segmenting Membranes",
        )
    finally:
        pool.shutdown()

    print("Completed the Membrane Segmentation!")


def run_segmenter(run, tomo_alg, voxel_size, session_id, threshold, gpu_id, models):
    from copick_utils.io import readers, writers

    from copick_torch.inference import membrain_seg

    # Default Sliding Window Parameters
    sw_batch_size = 4
    sw_window_size = 160

    # Read the Tomogram
    data = readers.tomogram(run, voxel_size, tomo_alg)

    # Segment the Tomogram
    predictions = membrain_seg.membrain_segment(
        data,
        models,
        sw_batch_size=sw_batch_size,
        sw_window_size=sw_window_size,
        test_time_augmentation=True,
        normalize_data=True,
        segmentation_threshold=threshold,
    )

    # Save the Segmentation
    writers.segmentation(run, predictions, "membrain-seg", "membranes", session_id=session_id, voxel_size=voxel_size)
