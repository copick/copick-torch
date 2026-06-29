"""CLI command for membrane segmentation with MemBrain-seg."""

import click
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger


@click.command(
    "membrain-seg",
    context_settings={"show_default": True},
    short_help="Segment membranes in tomograms with MemBrain-seg.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input tomograms.")
@optgroup.option(
    "--tomo-alg",
    "-ta",
    type=str,
    required=True,
    help="Tomogram algorithm/type to query and segment (e.g. 'wbp').",
)
@optgroup.option(
    "--voxel-size",
    "-vs",
    type=float,
    default=10.0,
    help="Voxel spacing to query, in angstroms.",
)
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--threshold",
    "-t",
    type=float,
    default=0.0,
    help="Segmentation threshold for the membrane probability map (0 keeps the raw map).",
)
@optgroup.group("\nOutput Options", help="Options related to the output segmentation.")
@optgroup.option(
    "--user-id",
    "-u",
    type=str,
    default="membrain-seg",
    help="User ID for the saved membrane segmentation.",
)
@optgroup.option(
    "--session-id",
    "-s",
    type=str,
    default="1",
    help="Session ID for the saved membrane segmentation.",
)
@add_debug_option
def membrain_seg(
    config: str,
    tomo_alg: str,
    voxel_size: float,
    threshold: float,
    user_id: str,
    session_id: str,
    debug: bool,
):
    """
    Segment membranes in tomograms with MemBrain-seg.

    For every run in the project, queries the tomogram of the given algorithm at the
    requested voxel spacing, runs the MemBrain-seg model with sliding-window inference
    (using test-time augmentation and input normalization), and writes the resulting
    membrane segmentation back into the project. Runs are processed in parallel on a
    GPU pool.

    A `threshold` of 0 stores the raw membrane probability map, while a positive
    threshold binarizes the prediction. The model weights are downloaded automatically
    on first use if they are not already cached. The output is saved as a segmentation
    named `membranes`.

    URI Format:

        \b
        Segmentations: name:user_id/session_id@voxel_spacing

    Examples:

        \b
        # Segment membranes in wbp tomograms at 10 A
        copick inference membrain-seg -c config.json --tomo-alg wbp --voxel-size 10.0

        \b
        # Binarize the prediction and tag the output user/session
        copick inference membrain-seg -c config.json --tomo-alg wbp --voxel-size 10.0 \\
            --threshold 0.5 --user-id membrain-seg --session-id 1

    See Also:

        \b
        copick process downsample: downsample tomograms before segmentation
    """
    run(config, tomo_alg, voxel_size, session_id, threshold, user_id, debug=debug)


def run(config, tomo_alg, voxel_size, session_id, threshold, user_id, debug=False):
    """
    Runs the membrane segmentation.
    """
    import copick

    from copick_torch import parallelization
    from copick_torch.inference import membrain_seg

    logger = get_logger(__name__, debug=debug)

    print("Starting Membrane Segmentation...")
    print(f"Using Tomograms with Voxel Size: {voxel_size} and Algorithm: {tomo_alg}")
    print(f"Saving Segmentations with {user_id}_{session_id}_membranes Query")
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

    tasks = [(run, tomo_alg, voxel_size, session_id, threshold, user_id) for run in root.runs]

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

    save_parameters(config, tomo_alg, voxel_size, session_id, user_id, threshold)
    logger.info("✅ Completed the Membrane Segmentation!")


def run_segmenter(run, tomo_alg, voxel_size, session_id, threshold, user_id, gpu_id, models):
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
    writers.segmentation(run, predictions, user_id, "membranes", session_id=session_id, voxel_size=voxel_size)


def save_parameters(config, tomo_alg, voxel_size, session_id, user_id, threshold):
    import os

    import copick

    from copick_torch.entry_points.utils import save_parameters_yaml

    # Determine Path to Save Parameters
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
        "parameters": {
            "threshold": threshold,
            "sw_batch_size": 4,
            "sw_window_size": 160,
        },
        "output": {
            "name": "membranes",
            "user-id": user_id,
            "session_id": session_id,
        },
    }
    os.makedirs(os.path.join(overlay_root, "logs"), exist_ok=True)
    path = os.path.join(overlay_root, "logs", f"segment-{user_id}_{session_id}_membranes.yaml")
    save_parameters_yaml(group, path)
    print(f"📝 Saved Parameters to {path}")
