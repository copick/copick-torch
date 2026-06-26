"""Fit surfaces to two pick sets and create a closed slab mesh."""

from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import tqdm
from copick.util.log import get_logger
from copick_utils.converters.converter_common import store_mesh_with_stats
from copick_utils.converters.lazy_converter import create_lazy_batch_converter
from copick_utils.converters.slab_common import triangulate_box

if TYPE_CHECKING:
    from copick.models import CopickMesh, CopickPicks, CopickRun

logger = get_logger(__name__)


def _bending_energy(grid) -> torch.Tensor:
    """Curvature (bending-energy) penalty for a CubicBSplineGrid2d.

    Computes the mean squared second difference of the spline control points along
    each grid axis and sums the two contributions. This directly penalizes
    curvature of the fitted surface; a larger weight on this term yields a flatter
    surface. Averaging (rather than summing) over control points keeps the penalty
    roughly invariant to ``grid_resolution`` so a given regularization weight
    behaves similarly across knot counts. Axes with fewer than 3 knots (no second
    difference) contribute zero.

    Args:
        grid: A ``CubicBSplineGrid2d`` whose control points live in ``grid._data``
            with shape ``(n_channels, rows, cols)``.

    Returns:
        Scalar tensor (on the grid's device) with the bending energy.
    """
    data = grid._data[0]  # (rows, cols) control-point heights
    penalty = data.sum() * 0.0  # zero scalar matching device/dtype
    if data.shape[0] >= 3:
        d2_rows = data[2:, :] - 2 * data[1:-1, :] + data[:-2, :]
        penalty = penalty + (d2_rows**2).mean()
    if data.shape[1] >= 3:
        d2_cols = data[:, 2:] - 2 * data[:, 1:-1] + data[:, :-2]
        penalty = penalty + (d2_cols**2).mean()
    return penalty


def fit_spline_surface(
    points: np.ndarray,
    max_dim: Sequence[float],
    grid_resolution: Tuple[int, int] = (5, 5),
    num_iterations: int = 500,
    learning_rate: float = 0.1,
    regularization: float = 0.0,
):
    """Fit a cubic B-spline surface to a set of 3D points.

    Args:
        points: Nx3 array of 3D point coordinates.
        max_dim: Physical dimensions [x, y, z] for normalization.
        grid_resolution: Resolution of the B-spline grid (rows, cols).
        num_iterations: Number of Adam optimizer iterations.
        learning_rate: Learning rate for the optimizer.
        regularization: Weight of the bending-energy (curvature) penalty. ``0.0``
            (default) reproduces the unregularized fit; higher values flatten the
            surface.

    Returns:
        Fitted CubicBSplineGrid2d object.
    """
    from torch_cubic_spline_grids import CubicBSplineGrid2d

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize xy to [0, 1] and z separately
    x = points[:, 0:2].copy()
    x[:, 0] /= max_dim[0]
    x[:, 1] /= max_dim[1]
    y = points[:, 2] / max_dim[2]

    grid = CubicBSplineGrid2d(resolution=tuple(grid_resolution), n_channels=1).to(device)
    optimizer = torch.optim.Adam(grid.parameters(), lr=learning_rate)

    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)

    t = tqdm.trange(num_iterations, desc="RMS: ", leave=True)
    for _ in t:
        pred = grid(x_t).squeeze()
        optimizer.zero_grad()
        data_loss = torch.sum((pred - y_t) ** 2) ** 0.5
        loss = data_loss + regularization * _bending_energy(grid) if regularization else data_loss
        loss.backward()
        optimizer.step()
        t.set_description(f"RMS: {data_loss.item():.6f}")
        t.refresh()

    return grid.cpu()


def evaluate_spline_on_grid(
    grid,
    fit_resolution: Tuple[int, int],
    max_dim: Sequence[float],
) -> np.ndarray:
    """Evaluate a fitted spline on a regular grid and return physical coordinates.

    Args:
        grid: Fitted CubicBSplineGrid2d object.
        fit_resolution: (rows, cols) grid resolution for evaluation.
        max_dim: Physical dimensions [x, y, z] for denormalization.

    Returns:
        Nx3 array of points in physical coordinates.
    """
    x_test, y_test = torch.meshgrid(
        torch.linspace(0, 1, fit_resolution[0]),
        torch.linspace(0, 1, fit_resolution[1]),
        indexing="xy",
    )
    xy = torch.stack([x_test, y_test], dim=2).view(-1, 2)

    points = np.empty((xy.shape[0], 3))
    points[:, 0] = xy[:, 0].numpy() * max_dim[0]
    points[:, 1] = xy[:, 1].numpy() * max_dim[1]
    points[:, 2] = grid(xy).squeeze().detach().numpy() * max_dim[2]

    return points


def fit_coupled_spline_slab(
    points1: np.ndarray,
    points2: np.ndarray,
    max_dim: Sequence[float],
    grid_resolution: Tuple[int, int] = (5, 5),
    num_iterations: int = 500,
    learning_rate: float = 0.1,
    regularization: float = 0.0,
):
    """Fit a curved-but-parallel slab to two point sets with a shared surface.

    A single cubic B-spline surface ``S(x, y)`` captures the common curvature of
    both layers, and two learned scalar z-offsets place the layers above/below it:
    ``top = S + off1``, ``bottom = S + off2``. The two surfaces therefore share one
    curvature and remain exactly parallel (constant vertical gap = ``off1 - off2``),
    in contrast to ``fit_spline_surface`` which fits each layer independently.

    Args:
        points1: Nx3 array of first surface points (e.g. top-layer).
        points2: Mx3 array of second surface points (e.g. bottom-layer).
        max_dim: Physical dimensions [x, y, z] for normalization.
        grid_resolution: Shared B-spline grid resolution (rows, cols).
        num_iterations: Number of Adam optimizer iterations.
        learning_rate: Learning rate for the optimizer.
        regularization: Weight of the bending-energy (curvature) penalty on the
            shared surface. Higher values flatten the slab.

    Returns:
        Tuple of (shared grid on CPU, off1, off2) where the offsets are CPU tensors
        in normalized z units.
    """
    from torch_cubic_spline_grids import CubicBSplineGrid2d

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize xy to [0, 1] and z separately
    xy1 = points1[:, 0:2].copy()
    xy1[:, 0] /= max_dim[0]
    xy1[:, 1] /= max_dim[1]
    z1 = points1[:, 2] / max_dim[2]

    xy2 = points2[:, 0:2].copy()
    xy2[:, 0] /= max_dim[0]
    xy2[:, 1] /= max_dim[1]
    z2 = points2[:, 2] / max_dim[2]

    grid = CubicBSplineGrid2d(resolution=tuple(grid_resolution), n_channels=1).to(device)
    # Initialize the offsets at each layer's mean height; the shared grid (init 0)
    # then represents the common deviation from those means.
    off1 = torch.nn.Parameter(torch.tensor([float(z1.mean())], device=device))
    off2 = torch.nn.Parameter(torch.tensor([float(z2.mean())], device=device))

    optimizer = torch.optim.Adam([*grid.parameters(), off1, off2], lr=learning_rate)

    xy1_t = torch.tensor(xy1, dtype=torch.float32, device=device)
    z1_t = torch.tensor(z1, dtype=torch.float32, device=device)
    xy2_t = torch.tensor(xy2, dtype=torch.float32, device=device)
    z2_t = torch.tensor(z2, dtype=torch.float32, device=device)

    t = tqdm.trange(num_iterations, desc="RMS: ", leave=True)
    for _ in t:
        optimizer.zero_grad()
        pred1 = grid(xy1_t).squeeze() + off1
        pred2 = grid(xy2_t).squeeze() + off2
        data_loss = torch.sum((pred1 - z1_t) ** 2) ** 0.5 + torch.sum((pred2 - z2_t) ** 2) ** 0.5
        loss = data_loss + regularization * _bending_energy(grid) if regularization else data_loss
        loss.backward()
        optimizer.step()
        t.set_description(f"RMS: {data_loss.item():.6f}")
        t.refresh()

    return grid.cpu(), off1.detach().cpu(), off2.detach().cpu()


def evaluate_coupled_on_grid(
    grid,
    offset,
    fit_resolution: Tuple[int, int],
    max_dim: Sequence[float],
) -> np.ndarray:
    """Evaluate a shared spline surface plus a z-offset on a regular grid.

    Mirrors ``evaluate_spline_on_grid`` but adds ``offset`` (in normalized z units)
    to the surface before denormalizing, so both layers of a coupled slab share the
    same xy sampling and curvature.

    Args:
        grid: Fitted shared CubicBSplineGrid2d object.
        offset: Scalar z-offset (normalized) for this layer.
        fit_resolution: (rows, cols) grid resolution for evaluation.
        max_dim: Physical dimensions [x, y, z] for denormalization.

    Returns:
        Nx3 array of points in physical coordinates.
    """
    x_test, y_test = torch.meshgrid(
        torch.linspace(0, 1, fit_resolution[0]),
        torch.linspace(0, 1, fit_resolution[1]),
        indexing="xy",
    )
    xy = torch.stack([x_test, y_test], dim=2).view(-1, 2)
    off = float(offset)

    points = np.empty((xy.shape[0], 3))
    points[:, 0] = xy[:, 0].numpy() * max_dim[0]
    points[:, 1] = xy[:, 1].numpy() * max_dim[1]
    points[:, 2] = (grid(xy).squeeze().detach().numpy() + off) * max_dim[2]

    return points


def _xy_to_z(xy: torch.Tensor, plane_normal: torch.Tensor, plane_offset: torch.Tensor) -> torch.Tensor:
    """Compute z coordinate from xy and plane parameters."""
    normal = plane_normal / torch.norm(plane_normal)
    d = torch.matmul(xy, normal[[2, 1]]) + plane_offset
    return -d / normal[0]


def fit_parallel_planes_from_picks(
    points1: np.ndarray,
    points2: np.ndarray,
    max_dim: Sequence[float],
    num_iterations: int = 500,
    learning_rate: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit two parallel planes to two sets of 3D points.

    Learns a shared plane normal and two offsets that minimize the RMS distance
    from each point set to its respective plane.

    Args:
        points1: Nx3 array of first surface points (e.g. top-layer).
        points2: Mx3 array of second surface points (e.g. bottom-layer).
        max_dim: Physical dimensions [x, y, z] for normalization.
        num_iterations: Number of Adam optimizer iterations.
        learning_rate: Learning rate for the optimizer.

    Returns:
        Tuple of (plane_normal, offset1, offset2) as CPU tensors.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize xy to [0, 1] and z separately
    xy1 = points1[:, 0:2].copy()
    xy1[:, 0] /= max_dim[0]
    xy1[:, 1] /= max_dim[1]
    z1 = points1[:, 2] / max_dim[2]

    xy2 = points2[:, 0:2].copy()
    xy2[:, 0] /= max_dim[0]
    xy2[:, 1] /= max_dim[1]
    z2 = points2[:, 2] / max_dim[2]

    xy1_t = torch.tensor(xy1, dtype=torch.float32, device=device)
    z1_t = torch.tensor(z1, dtype=torch.float32, device=device)
    xy2_t = torch.tensor(xy2, dtype=torch.float32, device=device)
    z2_t = torch.tensor(z2, dtype=torch.float32, device=device)

    # Learnable parameters: shared normal, two offsets
    plane_normal = torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0], device=device, requires_grad=True))
    offset1 = torch.nn.Parameter(torch.tensor([-0.5], device=device, requires_grad=True))
    offset2 = torch.nn.Parameter(torch.tensor([-0.5], device=device, requires_grad=True))

    optimizer = torch.optim.Adam([plane_normal, offset1, offset2], lr=learning_rate)

    t = tqdm.trange(num_iterations, desc="RMS: ", leave=True)
    for _ in t:
        optimizer.zero_grad()

        pred_z1 = _xy_to_z(xy1_t, plane_normal, offset1).squeeze()
        pred_z2 = _xy_to_z(xy2_t, plane_normal, offset2).squeeze()

        loss1 = torch.sum((pred_z1 - z1_t) ** 2) ** 0.5
        loss2 = torch.sum((pred_z2 - z2_t) ** 2) ** 0.5
        loss = loss1 + loss2

        loss.backward()
        optimizer.step()
        t.set_description(f"RMS: {loss.item():.6f}")
        t.refresh()

    return plane_normal.detach().cpu(), offset1.detach().cpu(), offset2.detach().cpu()


def evaluate_plane_on_grid(
    plane_normal: torch.Tensor,
    plane_offset: torch.Tensor,
    fit_resolution: Tuple[int, int],
    max_dim: Sequence[float],
) -> np.ndarray:
    """Evaluate a plane on a regular grid and return physical coordinates.

    Args:
        plane_normal: 3D plane normal vector.
        plane_offset: Scalar plane offset.
        fit_resolution: (rows, cols) grid resolution.
        max_dim: Physical dimensions [x, y, z] for denormalization.

    Returns:
        Nx3 array of points in physical coordinates.
    """
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, fit_resolution[1]),
        torch.linspace(0, 1, fit_resolution[0]),
        indexing="ij",
    )
    xy = torch.stack([yy, xx], dim=2).view(-1, 2)

    points = np.empty((xy.shape[0], 3))
    points[:, 0] = xy[:, 1].numpy() * max_dim[0]
    points[:, 1] = xy[:, 0].numpy() * max_dim[1]
    points[:, 2] = _xy_to_z(xy, plane_normal, plane_offset).squeeze().detach().numpy() * max_dim[2]

    return points


def _extract_pick_points(picks: "CopickPicks") -> np.ndarray:
    """Extract Nx3 point array from a CopickPicks object."""
    arr = np.empty((len(picks.points), 3))
    for i, p in enumerate(picks.points):
        arr[i, :] = [p.location.x, p.location.y, p.location.z]
    return arr


def slab_from_picks(
    picks1: "CopickPicks",
    picks2: "CopickPicks",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    tomo_type: str = "wbp",
    voxel_spacing: float = 10.0,
    method: str = "spline",
    grid_resolution: Tuple[int, int] = (5, 5),
    fit_resolution: Tuple[int, int] = (50, 50),
    num_iterations: int = 500,
    learning_rate: float = 0.1,
    regularization: float = 0.0,
    **kwargs,
) -> Optional[Tuple["CopickMesh", Dict[str, int]]]:
    """Create a closed slab mesh from two pick sets by fitting surfaces.

    Args:
        picks1: First set of picks (e.g. top-layer).
        picks2: Second set of picks (e.g. bottom-layer).
        run: CopickRun object.
        object_name: Name for the output mesh object.
        session_id: Session ID for the output mesh.
        user_id: User ID for the output mesh.
        tomo_type: Type of tomogram (for determining volume dimensions).
        voxel_spacing: Voxel spacing of the tomogram.
        method: Fitting method - "spline" for two independent B-spline surfaces,
            "coupled" for one shared curved surface with two offsets (curved but
            exactly parallel slab), "parallel" for two flat parallel planes (shared
            normal, two offsets).
        grid_resolution: B-spline grid resolution (rows, cols). Used with the
            "spline" and "coupled" methods (the knot grid).
        fit_resolution: Output mesh grid resolution (rows, cols).
        num_iterations: Number of optimizer iterations per surface.
        learning_rate: Learning rate for Adam optimizer.
        regularization: Bending-energy (curvature) penalty weight for the "spline"
            and "coupled" methods; higher = flatter. ``0.0`` (default) leaves the
            spline fit unregularized. Ignored for "parallel".

    Returns:
        Tuple of (CopickMesh, stats dict) or None if creation failed.
    """
    import zarr

    try:
        points1 = _extract_pick_points(picks1)
        points2 = _extract_pick_points(picks2)

        if len(points1) < 3 or len(points2) < 3:
            logger.warning(f"Need at least 3 points per surface, got {len(points1)} and {len(points2)}")
            return None

        # Get volume dimensions for normalization
        vs = run.get_voxel_spacing(voxel_spacing)
        tomo = vs.get_tomogram(tomo_type)
        shape = zarr.open(tomo.zarr())["0"].shape
        max_dim = [d * voxel_spacing for d in shape[::-1]]  # x, y, z in physical units

        if method == "parallel":
            logger.info(f"Fitting parallel planes to {len(points1)} + {len(points2)} points...")
            plane_normal, offset1, offset2 = fit_parallel_planes_from_picks(
                points1,
                points2,
                max_dim,
                num_iterations,
                learning_rate,
            )
            surface1 = evaluate_plane_on_grid(plane_normal, offset1, fit_resolution, max_dim)
            surface2 = evaluate_plane_on_grid(plane_normal, offset2, fit_resolution, max_dim)
        elif method == "coupled":
            logger.info(
                f"Fitting coupled (shared, parallel) slab to {len(points1)} + {len(points2)} points "
                f"(regularization={regularization})...",
            )
            grid, off1, off2 = fit_coupled_spline_slab(
                points1,
                points2,
                max_dim,
                grid_resolution,
                num_iterations,
                learning_rate,
                regularization,
            )
            surface1 = evaluate_coupled_on_grid(grid, off1, fit_resolution, max_dim)
            surface2 = evaluate_coupled_on_grid(grid, off2, fit_resolution, max_dim)
        else:
            logger.info(f"Fitting spline to {len(points1)} top-layer points (regularization={regularization})...")
            grid1 = fit_spline_surface(points1, max_dim, grid_resolution, num_iterations, learning_rate, regularization)

            logger.info(f"Fitting spline to {len(points2)} bottom-layer points (regularization={regularization})...")
            grid2 = fit_spline_surface(points2, max_dim, grid_resolution, num_iterations, learning_rate, regularization)

            surface1 = evaluate_spline_on_grid(grid1, fit_resolution, max_dim)
            surface2 = evaluate_spline_on_grid(grid2, fit_resolution, max_dim)

        # Create closed slab mesh
        mesh = triangulate_box(surface1, surface2, fit_resolution)

        return store_mesh_with_stats(run, mesh, object_name, session_id, user_id, "slab")

    except Exception as e:
        logger.error(f"Error creating slab mesh: {e}")
        return None


slab_from_picks_lazy_batch = create_lazy_batch_converter(
    converter_func=slab_from_picks,
    task_description="Fitting spline surfaces to picks and creating slab meshes",
)
