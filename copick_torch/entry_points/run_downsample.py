"""CLI command for downsampling tomograms with GPU Fourier rescaling."""

import click
from click_option_group import optgroup
from copick.cli.util import (
    add_config_option,
    add_debug_option,
    add_run_names_option,
    resolve_run_names,
    resolve_tomogram_uri,
)
from copick.util.log import get_logger
from copick.util.uri import parse_copick_uri
from copick_utils.cli.util import add_input_option, add_output_option


@click.command(
    "downsample",
    context_settings={"show_default": True},
    short_help="Downsample tomograms via Fourier rescaling.",
    no_args_is_help=True,
)
@add_config_option
@add_run_names_option
@optgroup.group("\nInput Options", help="Options related to the input tomograms.")
@add_input_option("tomogram", required=False)
# TODO:remove once deprecation takes effect -- legacy --tomo-alg/--voxel-size fallback
@optgroup.option(
    "--tomo-alg",
    "-ta",
    type=str,
    default=None,
    hidden=True,
    help="Deprecated: use -i/--input <tomo_type>@<voxel_spacing>.",
)
@optgroup.option(
    "--voxel-size",
    "-vs",
    type=float,
    default=None,
    hidden=True,
    help="Deprecated: use -i/--input <tomo_type>@<voxel_spacing>.",
)
@optgroup.group("\nOutput Options", help="Options related to the output tomograms.")
@add_output_option("tomogram", required=False)
# TODO:remove once deprecation takes effect -- legacy --target-resolution fallback
@optgroup.option(
    "--target-resolution",
    "-tr",
    type=float,
    default=None,
    hidden=True,
    help="Deprecated: use -o/--output <tomo_type>@<voxel_spacing>.",
)
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--delete-source/--keep-source",
    is_flag=True,
    default=False,
    help="Delete the source tomograms after downsampling.",
)
@add_debug_option
def downsample(
    config: str,
    run_names,
    input_uri,
    tomo_alg,
    voxel_size,
    output_uri,
    target_resolution,
    delete_source: bool,
    debug: bool,
):
    """
    Downsample tomograms on the GPU with Fourier rescaling.

    For every run in the project (or only the runs given by --run-names/-r), queries
    the input tomogram, Fourier-rescales it to the output voxel spacing on the GPU, and
    writes the downsampled tomogram back into the project. The output voxel spacing (and
    optional rename) is taken from -o/--output. Runs are processed in parallel on a GPU pool.

    URI Format:

        \b
        Tomograms: tomo_type@voxel_spacing

    Examples:

        \b
        # Downsample wbp tomograms from 10 A to 20 A
        copick process downsample -c config.json -i wbp@10.0 -o wbp@20.0

        \b
        # Downsample and rename the output, then delete the source tomograms
        copick process downsample -c config.json -i wbp@10.0 -o wbp-bin2@20.0 --delete-source

    See Also:

        \b
        copick process bandpass: band-pass filter tomograms without resampling
    """
    logger = get_logger(__name__, debug=debug)

    in_uri = resolve_tomogram_uri(input_uri, tomo_alg, voxel_size, logger=logger)
    in_params = parse_copick_uri(in_uri, "tomogram")
    in_tomo = in_params["tomo_type"]
    in_vs = float(in_params["voxel_spacing"])

    if output_uri:
        out_params = parse_copick_uri(output_uri, "tomogram")
        out_tomo = out_params["tomo_type"]
        out_vs = float(out_params["voxel_spacing"])
    # TODO:remove once deprecation takes effect -- legacy --target-resolution fallback
    elif target_resolution is not None:
        logger.warning("--target-resolution is deprecated; use -o/--output <tomo_type>@<voxel_spacing>.")
        out_tomo = in_tomo
        out_vs = target_resolution
    else:
        raise click.UsageError("Provide the output via -o/--output <tomo_type>@<voxel_spacing>.")

    run(
        config,
        in_tomo,
        in_vs,
        out_vs,
        delete_source,
        write_algorithm=out_tomo,
        run_names=run_names,
        debug=debug,
    )


def run(
    config,
    tomo_alg,
    voxel_size,
    target_resolution,
    delete_source,
    write_algorithm=None,
    run_names=None,
    debug=False,
):
    """
    Runs the downsampling.
    """

    import copick
    from copick.ops.get import get_runs

    from copick_torch import parallelization
    from copick_torch.filters import downsample

    logger = get_logger(__name__, debug=debug)

    write_algorithm = write_algorithm or tomo_alg

    root = copick.from_file(config)
    run_names_list = resolve_run_names(run_names, logger=logger)
    runs = get_runs(root, run_names_list)
    run_ids = [r.name for r in runs]

    pool = parallelization.GPUPool(
        init_fn=downsample.downsample_init,
        init_args=(voxel_size, target_resolution),
        verbose=True,
    )

    tasks = [(r, tomo_alg, voxel_size, target_resolution, delete_source, write_algorithm) for r in runs]

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
    logger.info("✅ Completed the Downsampling!")


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
