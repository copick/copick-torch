"""Fit cubic B-spline surfaces to two pick sets and create a closed slab mesh."""

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


def fit_spline_surface(
    points: np.ndarray,
    max_dim: Sequence[float],
    grid_resolution: Tuple[int, int] = (5, 5),
    num_iterations: int = 500,
    learning_rate: float = 0.1,
):
    """Fit a cubic B-spline surface to a set of 3D points.

    Args:
        points: Nx3 array of 3D point coordinates.
        max_dim: Physical dimensions [x, y, z] for normalization.
        grid_resolution: Resolution of the B-spline grid (rows, cols).
        num_iterations: Number of Adam optimizer iterations.
        learning_rate: Learning rate for the optimizer.

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
        loss = torch.sum((pred - y_t) ** 2) ** 0.5
        loss.backward()
        optimizer.step()
        t.set_description(f"RMS: {loss.item():.6f}")
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
    grid_resolution: Tuple[int, int] = (5, 5),
    fit_resolution: Tuple[int, int] = (50, 50),
    num_iterations: int = 500,
    learning_rate: float = 0.1,
    **kwargs,
) -> Optional[Tuple["CopickMesh", Dict[str, int]]]:
    """Create a closed slab mesh from two pick sets by fitting B-spline surfaces.

    Args:
        picks1: First set of picks (e.g. top-layer).
        picks2: Second set of picks (e.g. bottom-layer).
        run: CopickRun object.
        object_name: Name for the output mesh object.
        session_id: Session ID for the output mesh.
        user_id: User ID for the output mesh.
        tomo_type: Type of tomogram (for determining volume dimensions).
        voxel_spacing: Voxel spacing of the tomogram.
        grid_resolution: B-spline grid resolution (rows, cols).
        fit_resolution: Output mesh grid resolution (rows, cols).
        num_iterations: Number of optimizer iterations per surface.
        learning_rate: Learning rate for Adam optimizer.

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

        logger.info(f"Fitting spline to {len(points1)} top-layer points...")
        grid1 = fit_spline_surface(points1, max_dim, grid_resolution, num_iterations, learning_rate)

        logger.info(f"Fitting spline to {len(points2)} bottom-layer points...")
        grid2 = fit_spline_surface(points2, max_dim, grid_resolution, num_iterations, learning_rate)

        # Evaluate on regular grids
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
