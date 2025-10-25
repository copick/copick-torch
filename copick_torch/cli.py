import click

import copick_torch
from copick_torch.entry_points.run_membrane_seg import membrain_seg
from copick_torch.entry_points.run_downsample import downsample
from copick_torch.entry_points.run_filter3d import bandpass


@click.group()
def routines():
    pass

routines.add_command(membrain_seg)
routines.add_command(downsample)
routines.add_command(bandpass)

if __name__ == "__main__":
    routines()
