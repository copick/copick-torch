"""CLI command for fitting parallel planes to a segmentation and creating a slab mesh."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger
from copick.util.uri import parse_copick_uri
from copick_utils.cli.util import add_input_option, add_output_option, add_workers_option
from copick_utils.util.config_models import create_simple_config


@click.command(
    "seg2slab",
    context_settings={"show_default": True},
    short_help="Fit parallel planes to a segmentation and create a slab mesh.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input segmentations.")
@optgroup.option(
    "--run-names",
    "-r",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@add_input_option("segmentation")
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--label",
    "-l",
    type=int,
    required=True,
    help="Label index to extract from the segmentation.",
)
@optgroup.option(
    "--fit-resolution",
    nargs=2,
    type=int,
    default=(50, 50),
    help="Output mesh grid resolution (rows cols).",
)
@optgroup.option(
    "--num-iterations",
    type=int,
    default=20,
    help="Number of optimization iterations for plane fitting.",
)
@optgroup.option(
    "--learning-rate",
    type=float,
    default=0.1,
    help="Learning rate for Adam optimizer.",
)
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output meshes.")
@add_output_option("mesh", default_tool="seg2slab")
@add_debug_option
def seg2slab(
    config,
    run_names,
    input_uri,
    label,
    fit_resolution,
    num_iterations,
    learning_rate,
    workers,
    output_uri,
    debug,
):
    """
    Fit parallel planes to a segmentation volume and create a closed slab mesh.

    Extracts a single label from the segmentation, finds the largest connected
    component, then fits two parallel planes (shared normal, different offsets)
    by optimizing IoU. The fitted planes are connected with side walls to form
    a closed, watertight box mesh.

    \b
    URI Format:
        Segmentations: name:user_id/session_id@voxel_spacing
        Meshes: object_name:user_id/session_id

    \b
    Examples:
        # Fit slab from a segmentation
        copick convert seg2slab -c config.json \\
            -i "segmentation:output/0@7.84" \\
            --label 2 \\
            -o "valid-sample:seg2slab/0"

        # Process specific runs with custom resolution
        copick convert seg2slab -c config.json \\
            -r 14114 -r 14132 \\
            -i "predictions:model/run-001@10.0" \\
            --label 1 --fit-resolution 100 100 \\
            -o "sample:seg2slab/fitted"
    """
    from copick_torch.fitting.slab_from_segmentation import slab_from_segmentation_lazy_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Create simple config for segmentation → mesh
    try:
        task_config = create_simple_config(
            input_uri=input_uri,
            input_type="segmentation",
            output_uri=output_uri,
            output_type="mesh",
            command_name="seg2slab",
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Log parameters
    input_params = parse_copick_uri(input_uri, "segmentation")
    output_params = parse_copick_uri(output_uri, "mesh")

    logger.info(f"Fitting parallel planes to segmentation '{input_params['name']}'")
    logger.info(f"Source: {input_params['user_id']}/{input_params['session_id']}")
    logger.info(
        f"Target mesh: {output_params['object_name']} ({output_params['user_id']}/{output_params['session_id']})",
    )
    logger.info(f"Label: {label}, Fit resolution: {fit_resolution}")

    # Run batch conversion
    results = slab_from_segmentation_lazy_batch(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
        label=label,
        fit_resolution=tuple(fit_resolution),
        num_iterations=num_iterations,
        learning_rate=learning_rate,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_vertices = sum(result.get("vertices_created", 0) for result in results.values() if result)
    total_faces = sum(result.get("faces_created", 0) for result in results.values() if result)
    total_processed = sum(result.get("processed", 0) for result in results.values() if result)

    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total slab meshes created: {total_processed}")
    logger.info(f"Total vertices created: {total_vertices}")
    logger.info(f"Total faces created: {total_faces}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
