import click

_cli_context = {
    "show_default": True,
    "help_option_names": ["-h", "--help"],
}


@click.group("nnunet", context_settings=_cli_context, no_args_is_help=True)
def cli():
    """nnUNet integration for CoPick cryo-ET datasets."""
    pass


from copick_torch.nnunet.prepare import cli as prepare_cli  # noqa: E402
from copick_torch.nnunet.train import cli as train_cli      # noqa: E402
from copick_torch.nnunet.predict import cli as predict_cli  # noqa: E402

cli.add_command(prepare_cli, "prepare")
cli.add_command(train_cli, "train")
cli.add_command(predict_cli, "segment")
