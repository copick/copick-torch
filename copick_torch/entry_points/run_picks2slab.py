"""CLI command for fitting B-spline surfaces to two pick sets and creating a slab mesh."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger
from copick.util.uri import parse_copick_uri

from copick_utils.cli.util import add_dual_input_options, add_output_option, add_tomogram_option, add_workers_option
from copick_utils.util.config_models import create_dual_selector_config


@click.command(
    "picks2slab",
    context_settings={"show_default": True},
    short_help="Fit spline surfaces to two pick sets and create a slab mesh.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input picks.")
@optgroup.option(
    "--run-names",
    "-r",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@add_dual_input_options("picks")
@optgroup.group("\nTool Options", help="Options related to this tool.")
@add_tomogram_option(required=True)
@optgroup.option(
    "--grid-resolution",
    nargs=2,
    type=int,
    default=(5, 5),
    help="B-spline grid resolution (rows cols).",
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
    default=500,
    help="Number of optimization iterations per surface.",
)
@optgroup.option(
    "--learning-rate",
    type=float,
    default=0.1,
    help="Learning rate for Adam optimizer.",
)
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output meshes.")
@add_output_option("mesh", default_tool="picks2slab")
@add_debug_option
def picks2slab(
    config,
    run_names,
    input1_uri,
    input2_uri,
    tomogram_uri,
    grid_resolution,
    fit_resolution,
    num_iterations,
    learning_rate,
    workers,
    output_uri,
    debug,
):
    """
    Fit cubic B-spline surfaces to two pick sets and create a closed slab mesh.

    Takes two sets of picks (e.g. top-layer and bottom-layer boundary annotations)
    and fits independent cubic B-spline surfaces to each. The fitted surfaces are
    then connected with side walls to form a closed, watertight slab mesh.

    \b
    URI Format:
        Picks: object_name:user_id/session_id
        Meshes: object_name:user_id/session_id
        Tomograms: tomo_type@voxel_spacing

    \b
    Examples:
        # Fit slab from top and bottom layer picks
        copick convert picks2slab -c config.json \\
            -i1 "top-layer:bob/1" -i2 "bottom-layer:bob/1" \\
            -t "wbp@7.84" \\
            -o "sample:picks2slab/0"

        # With custom spline and mesh resolution
        copick convert picks2slab -c config.json \\
            -i1 "top-layer:user1/manual" -i2 "bottom-layer:user1/manual" \\
            -t "wbp@10.0" \\
            --grid-resolution 7 7 --fit-resolution 100 100 \\
            -o "sample:picks2slab/fitted"
    """
    from copick_torch.fitting.slab_from_picks import slab_from_picks_lazy_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Parse tomogram URI to extract tomo_type and voxel_spacing
    tomo_params = parse_copick_uri(tomogram_uri, "tomogram")
    tomo_type = tomo_params["tomo_type"]
    voxel_spacing = tomo_params["voxel_spacing"]

    # Create dual-selector config for picks → mesh
    try:
        task_config = create_dual_selector_config(
            input1_uri=input1_uri,
            input2_uri=input2_uri,
            input_type="picks",
            output_uri=output_uri,
            output_type="mesh",
            command_name="picks2slab",
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Log parameters
    input1_params = parse_copick_uri(input1_uri, "picks")
    input2_params = parse_copick_uri(input2_uri, "picks")
    output_params = parse_copick_uri(output_uri, "mesh")

    logger.info(f"Fitting slab from '{input1_params['object_name']}' and '{input2_params['object_name']}'")
    logger.info(f"Input 1: {input1_params['user_id']}/{input1_params['session_id']}")
    logger.info(f"Input 2: {input2_params['user_id']}/{input2_params['session_id']}")
    logger.info(
        f"Target mesh: {output_params['object_name']} ({output_params['user_id']}/{output_params['session_id']})"
    )
    logger.info(f"Tomogram: {tomo_type}@{voxel_spacing}")
    logger.info(f"Grid resolution: {grid_resolution}, Fit resolution: {fit_resolution}")

    # Run batch conversion
    results = slab_from_picks_lazy_batch(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
        tomo_type=tomo_type,
        voxel_spacing=voxel_spacing,
        grid_resolution=tuple(grid_resolution),
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
