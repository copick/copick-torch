"""Tests for spline/coupled slab fitting in copick_torch.fitting.slab_from_picks.

These exercise the surface-fitting math directly (no copick I/O): the bending-energy
regularizer, its effect on the independent ``spline`` fit, and the curved-but-parallel
``coupled`` fit. They back the curvature-regularization and coupled-mode additions to
``copick convert picks2slab``.
"""

import numpy as np
import torch
from torch_cubic_spline_grids import CubicBSplineGrid2d

from copick_torch.fitting.slab_from_picks import (
    _bending_energy,
    evaluate_coupled_on_grid,
    fit_coupled_spline_slab,
    fit_spline_surface,
)
from copick_torch.fitting.slab_from_segmentation import _extract_surface_points


def _set_control_points(grid, values_2d: np.ndarray) -> None:
    """Overwrite a CubicBSplineGrid2d's control points with ``values_2d`` (rows, cols)."""
    with torch.no_grad():
        grid._data.copy_(torch.tensor(values_2d, dtype=grid._data.dtype).reshape(grid._data.shape))


def _fit_rms(grid, points: np.ndarray, max_dim) -> float:
    """RMS (Angstrom) of a fitted spline grid against the picks it was fit to."""
    xy = points[:, 0:2].copy()
    xy[:, 0] /= max_dim[0]
    xy[:, 1] /= max_dim[1]
    pred = grid(torch.tensor(xy, dtype=torch.float32)).squeeze().detach().numpy() * max_dim[2]
    return float(np.sqrt(np.mean((pred - points[:, 2]) ** 2)))


def test_bending_energy_zero_for_affine_grid():
    """An affine (planar) control grid has no curvature; a quadratic one does."""
    r, c = 6, 5
    ii, jj = np.meshgrid(np.arange(r), np.arange(c), indexing="ij")

    affine = 0.3 * ii - 0.2 * jj + 1.0  # zero second differences along both axes
    grid_flat = CubicBSplineGrid2d(resolution=(r, c), n_channels=1)
    _set_control_points(grid_flat, affine)
    assert float(_bending_energy(grid_flat).detach()) < 1e-10

    curved = (ii.astype(float)) ** 2  # constant nonzero second difference along rows
    grid_curved = CubicBSplineGrid2d(resolution=(r, c), n_channels=1)
    _set_control_points(grid_curved, curved)
    assert float(_bending_energy(grid_curved).detach()) > 1e-3


def test_spline_unregularized_fits_plane():
    """With regularization=0 (default), the spline fit still tracks the data closely."""
    torch.manual_seed(1)
    np.random.seed(1)
    max_dim = [800.0, 800.0, 400.0]
    n = 120
    xy = np.random.rand(n, 2)
    pts = np.empty((n, 3))
    pts[:, 0] = xy[:, 0] * max_dim[0]
    pts[:, 1] = xy[:, 1] * max_dim[1]
    pts[:, 2] = 200.0 + 50.0 * xy[:, 0] - 30.0 * xy[:, 1]  # tilted plane

    grid = fit_spline_surface(pts, max_dim, (5, 5), num_iterations=400, learning_rate=0.1, regularization=0.0)
    assert _fit_rms(grid, pts, max_dim) < 10.0


def test_spline_regularization_reduces_curvature():
    """A larger regularization weight yields a strictly flatter fitted surface."""
    torch.manual_seed(0)
    np.random.seed(0)
    max_dim = [1000.0, 1000.0, 500.0]
    n = 150
    xy = np.random.rand(n, 2)
    bump = 120.0 * np.sin(2 * np.pi * xy[:, 0]) * np.sin(2 * np.pi * xy[:, 1])
    pts = np.empty((n, 3))
    pts[:, 0] = xy[:, 0] * max_dim[0]
    pts[:, 1] = xy[:, 1] * max_dim[1]
    pts[:, 2] = 250.0 + bump

    grid_0 = fit_spline_surface(pts, max_dim, (7, 7), num_iterations=250, learning_rate=0.1, regularization=0.0)
    grid_reg = fit_spline_surface(pts, max_dim, (7, 7), num_iterations=250, learning_rate=0.1, regularization=200.0)

    assert float(_bending_energy(grid_reg).detach()) < float(_bending_energy(grid_0).detach())


def test_coupled_surfaces_are_parallel():
    """The coupled fit produces two surfaces with a constant gap (exactly parallel)."""
    torch.manual_seed(0)
    np.random.seed(0)
    max_dim = [1000.0, 1000.0, 500.0]
    n = 200
    gap = 100.0
    xy = np.random.rand(n, 2)
    bump = 80.0 * np.sin(2 * np.pi * xy[:, 0])

    def make(offset):
        p = np.empty((n, 3))
        p[:, 0] = xy[:, 0] * max_dim[0]
        p[:, 1] = xy[:, 1] * max_dim[1]
        p[:, 2] = 250.0 + bump + offset
        return p

    top = make(gap / 2)
    bottom = make(-gap / 2)

    grid, off1, off2 = fit_coupled_spline_slab(
        top,
        bottom,
        max_dim,
        (5, 5),
        num_iterations=300,
        learning_rate=0.1,
        regularization=0.0,
    )
    surf1 = evaluate_coupled_on_grid(grid, off1, (40, 40), max_dim)
    surf2 = evaluate_coupled_on_grid(grid, off2, (40, 40), max_dim)
    vertical_gap = surf1[:, 2] - surf2[:, 2]

    # Constant gap everywhere -> the two surfaces are parallel.
    assert vertical_gap.std() < 0.5
    # The shared-surface + offsets construction makes the gap equal off1 - off2.
    assert abs((float(off1) - float(off2)) * max_dim[2] - vertical_gap.mean()) < 1e-2
    # And it recovers the true slab thickness.
    assert abs(vertical_gap.mean() - gap) < 10.0


def test_extract_surface_points_picks_min_max_z():
    """Surface extraction returns the first/last foreground voxel per column, in Angstrom."""
    vs = 10.0
    vol = np.zeros((20, 30, 30), np.uint8)
    vol[4:10, 5:25, 5:25] = 1  # slab occupies z indices 4..9 over a 20x20 xy region

    top, bot = _extract_surface_points(vol, voxel_size=vs, stride=1)

    assert len(top) == len(bot) == 20 * 20  # one point per foreground column
    # Last foreground index along z is 9, first is 4 -> physical z = idx * vs.
    assert np.allclose(np.unique(top[:, 2]), [9 * vs])
    assert np.allclose(np.unique(bot[:, 2]), [4 * vs])
    # xy span follows the foreground footprint (indices 5..24).
    assert top[:, 0].min() == 5 * vs and top[:, 0].max() == 24 * vs
    assert top[:, 1].min() == 5 * vs and top[:, 1].max() == 24 * vs

    # Striding decimates columns but preserves the surface heights.
    ts, bs = _extract_surface_points(vol, voxel_size=vs, stride=3)
    assert 0 < len(ts) < len(top)
    assert np.allclose(np.unique(ts[:, 2]), [9 * vs])


def test_coupled_fit_on_extracted_surface_points():
    """A curved binary slab -> extracted surfaces -> coupled fit recovers a constant gap."""
    torch.manual_seed(0)
    vs = 10.0
    nz, ny, nx = 60, 40, 40
    vol = np.zeros((nz, ny, nx), np.uint8)
    gap = 12  # voxels
    for ix in range(nx):
        center = 30 + int(round(6 * np.sin(2 * np.pi * ix / nx)))  # curved slab center
        z0, z1 = center - gap // 2, center + gap // 2
        vol[z0:z1, :, ix] = 1

    top, bot = _extract_surface_points(vol, voxel_size=vs, stride=1)
    max_dim = [nx * vs, ny * vs, nz * vs]

    grid, off1, off2 = fit_coupled_spline_slab(
        top,
        bot,
        max_dim,
        (5, 5),
        num_iterations=300,
        learning_rate=0.1,
        regularization=0.0,
    )
    surf1 = evaluate_coupled_on_grid(grid, off1, (40, 40), max_dim)
    surf2 = evaluate_coupled_on_grid(grid, off2, (40, 40), max_dim)
    vertical_gap = surf1[:, 2] - surf2[:, 2]

    # Exactly-parallel surfaces -> near-constant gap, recovering the ~gap-voxel thickness.
    assert vertical_gap.std() < 5.0
    assert abs(abs(vertical_gap.mean()) - gap * vs) < 25.0
