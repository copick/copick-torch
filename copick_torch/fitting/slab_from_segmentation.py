"""Fit a slab mesh to a segmentation volume.

Two families of fitting are supported:

* the ``"spline"`` / ``"coupled"`` / ``"parallel"`` methods extract top- and bottom-surface
  point-clouds from the (largest connected component of the) segmentation and reuse the tested
  fitters from :mod:`copick_torch.fitting.slab_from_picks`, giving the same options as
  ``picks2slab`` (B-spline grid resolution, curvature regularization);
* the legacy ``"iou"`` method fits two flat parallel planes directly to the binary volume by
  maximizing a differentiable intersection-over-union.

All methods produce a closed, watertight box mesh.
"""

from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import tqdm
from copick.util.log import get_logger
from copick_utils.converters.converter_common import store_mesh_with_stats
from copick_utils.converters.lazy_converter import create_lazy_batch_converter
from copick_utils.converters.slab_common import triangulate_box
from skimage import measure

from copick_torch.fitting.slab_from_picks import (
    evaluate_coupled_on_grid,
    evaluate_spline_on_grid,
    fit_coupled_spline_slab,
    fit_parallel_planes_from_picks,
    fit_spline_surface,
)
from copick_torch.fitting.slab_from_picks import (
    evaluate_plane_on_grid as evaluate_plane_on_grid_picks,
)

if TYPE_CHECKING:
    from copick.models import CopickMesh, CopickRun, CopickSegmentation

logger = get_logger(__name__)


def get_largest_component(volume: np.ndarray) -> np.ndarray:
    """Extract the largest connected component from a binary volume.

    Args:
        volume: Binary 3D volume.

    Returns:
        Binary volume containing only the largest component.
    """
    labeled = measure.label(volume)
    props = measure.regionprops(labeled)
    if not props:
        return volume
    largest = max(props, key=lambda x: x.area)
    out = np.zeros_like(volume)
    out[labeled == largest.label] = 1
    return out


def _extract_surface_points(
    binary: np.ndarray,
    voxel_size: float,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract top- and bottom-surface point clouds from a binary slab volume.

    For every ``(y, x)`` column that contains foreground, the topmost (max-z) and bottommost
    (min-z) foreground voxels define one point on the top and bottom surface respectively. The
    slab normal is assumed to be roughly the z-axis (axis 0), matching the picks-based fitters
    which model each surface as a height field ``z = f(x, y)``.

    Args:
        binary: Binary 3D volume (z, y, x).
        voxel_size: Voxel spacing in angstroms (to convert voxel indices to physical coordinates).
        stride: Column subsampling stride (>=1) to bound the point count on large volumes.

    Returns:
        Tuple ``(top_points, bottom_points)``, each an Nx3 array of ``[x, y, z]`` physical
        (angstrom) coordinates following the convention used by the picks fitters.
    """
    fg = binary > 0
    nz = fg.shape[0]

    # First / last foreground voxel along z per column (argmax over a reversed view = last True).
    bot_z = fg.argmax(axis=0)  # first foreground (0 for empty columns, masked out below)
    top_z = nz - 1 - fg[::-1].argmax(axis=0)  # last foreground
    any_fg = fg.any(axis=0)

    if stride > 1:
        keep = np.zeros_like(any_fg)
        keep[::stride, ::stride] = True
        any_fg = any_fg & keep

    ys, xs = np.nonzero(any_fg)
    top_points = np.stack([xs * voxel_size, ys * voxel_size, top_z[ys, xs] * voxel_size], axis=1).astype(float)
    bot_points = np.stack([xs * voxel_size, ys * voxel_size, bot_z[ys, xs] * voxel_size], axis=1).astype(float)
    return top_points, bot_points


def _xy_to_z(xy: torch.Tensor, plane_normal: torch.Tensor, plane_offset: torch.Tensor) -> torch.Tensor:
    """Compute z coordinate from xy and plane parameters."""
    normal = plane_normal / torch.norm(plane_normal)
    d = torch.matmul(xy, normal[[2, 1]]) + plane_offset
    return -d / normal[0]


def fit_parallel_planes(
    volume: np.ndarray,
    num_iterations: int = 20,
    learning_rate: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit two parallel planes to a binary segmentation volume using differentiable IoU.

    Learns a shared plane normal and two offsets (top, bottom) that maximize the
    intersection-over-union between the plane-defined slab and the input volume.

    Args:
        volume: Binary 3D volume (z, y, x).
        num_iterations: Number of Adam optimizer iterations.
        learning_rate: Learning rate for the optimizer.

    Returns:
        Tuple of (plane_normal, top_offset, bot_offset) as CPU tensors.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vol = torch.tensor(volume, dtype=torch.float32, device=device)
    voldim = torch.tensor(vol.shape, dtype=torch.int32, device=device)

    # Grid for evaluating the spline (2D xy)
    yy2, xx2 = torch.meshgrid(
        torch.arange(0, vol.shape[1], 1),
        torch.arange(0, vol.shape[2], 1),
        indexing="ij",
    )

    # 3D grid for z-distance computation
    zz, _, _ = torch.meshgrid(
        torch.arange(0, vol.shape[0], 1),
        torch.arange(0, vol.shape[1], 1),
        torch.arange(0, vol.shape[2], 1),
        indexing="ij",
    )

    # Normalize to [0, 1]
    xx2 = xx2.to(device) / voldim[2]
    yy2 = yy2.to(device) / voldim[1]
    zz = zz.to(device) / voldim[0]
    xy2 = torch.stack([yy2, xx2], dim=2).view(-1, 2).to(device)

    # Learnable parameters
    plane_normal = torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0], device=device, requires_grad=True))
    top_offset = torch.nn.Parameter(torch.tensor([-0.8], device=device, requires_grad=True))
    bot_offset = torch.nn.Parameter(torch.tensor([-0.2], device=device, requires_grad=True))

    optimizer = torch.optim.Adam([plane_normal, top_offset, bot_offset], lr=learning_rate)

    t = tqdm.trange(num_iterations, desc="1 - IoU: ", leave=True)

    for _ in t:
        optimizer.zero_grad()

        zz_top = _xy_to_z(xy2, plane_normal, top_offset).squeeze().reshape((voldim[1], voldim[2])).to(device)
        zz_bot = _xy_to_z(xy2, plane_normal, bot_offset).squeeze().reshape((voldim[1], voldim[2])).to(device)

        # Differentiable slab mask using steep sigmoid
        valid = torch.sigmoid(1000 * (zz_top - zz)) * torch.sigmoid(1000 * (zz - zz_bot))

        intersection = torch.sum(valid * vol)
        union = torch.sum(valid) + torch.sum(vol) - intersection
        loss = 1 - intersection / union

        loss.backward()
        optimizer.step()
        t.set_description(f"1 - IoU: {loss.item():.6f}")
        t.refresh()

    return plane_normal.detach().cpu(), top_offset.detach().cpu(), bot_offset.detach().cpu()


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


def slab_from_segmentation(
    segmentation: "CopickSegmentation",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    label: int = 1,
    method: str = "coupled",
    grid_resolution: Tuple[int, int] = (5, 5),
    fit_resolution: Tuple[int, int] = (50, 50),
    num_iterations: int = 500,
    learning_rate: float = 0.1,
    regularization: float = 0.0,
    surface_stride: int = 1,
    **kwargs,
) -> Optional[Tuple["CopickMesh", Dict[str, int]]]:
    """Create a closed slab mesh by fitting a surface to a segmentation.

    Extracts a single label, keeps its largest connected component, then fits a slab using one
    of several methods and connects the two surfaces into a closed, watertight box mesh.

    Args:
        segmentation: CopickSegmentation object.
        run: CopickRun object.
        object_name: Name for the output mesh object.
        session_id: Session ID for the output mesh.
        user_id: User ID for the output mesh.
        label: Label index to extract from the segmentation.
        method: Fitting method. ``"spline"`` fits two independent B-spline surfaces to the
            extracted top/bottom surface points; ``"coupled"`` fits one shared curved surface
            with two offsets (curved but exactly parallel slab); ``"parallel"`` fits two flat
            parallel planes to the surface points; ``"iou"`` fits two flat parallel planes
            directly to the binary volume by maximizing intersection-over-union (legacy).
        grid_resolution: B-spline knot grid resolution (rows, cols) for ``spline``/``coupled``.
        fit_resolution: Output mesh grid resolution (rows, cols).
        num_iterations: Number of optimizer iterations.
        learning_rate: Learning rate for Adam optimizer.
        regularization: Bending-energy (curvature) penalty weight for ``spline``/``coupled``;
            higher = flatter. Ignored for ``parallel``/``iou``.
        surface_stride: Column subsampling stride for surface-point extraction (>=1); bounds the
            point count on large volumes. Ignored for ``iou``.

    Returns:
        Tuple of (CopickMesh, stats dict) or None if creation failed.
    """
    try:
        # Load and extract label
        vol = segmentation.numpy()

        if vol is None or vol.size == 0:
            logger.error("Empty or invalid volume")
            return None

        binary = np.zeros_like(vol)
        binary[vol == label] = 1

        if binary.sum() == 0:
            logger.warning(f"No voxels found with label {label}")
            return None

        # Get largest connected component
        logger.info("Extracting largest connected component...")
        binary = get_largest_component(binary)

        # Compute physical dimensions
        voxel_size = segmentation.voxel_size
        max_dim = [d * voxel_size for d in vol.shape[::-1]]  # x, y, z

        method = (method or "coupled").lower()

        if method == "iou":
            # Legacy: fit flat parallel planes directly to the volume via IoU.
            logger.info("Fitting parallel planes to segmentation (IoU)...")
            plane_normal, top_offset, bot_offset = fit_parallel_planes(binary, num_iterations, learning_rate)
            surface1 = evaluate_plane_on_grid(plane_normal, top_offset, fit_resolution, max_dim)
            surface2 = evaluate_plane_on_grid(plane_normal, bot_offset, fit_resolution, max_dim)
        else:
            # Extract top/bottom surface point-clouds and reuse the picks-based fitters.
            top_points, bot_points = _extract_surface_points(binary, voxel_size, stride=surface_stride)
            if len(top_points) < 3 or len(bot_points) < 3:
                logger.warning(
                    f"Need at least 3 surface points per layer, got {len(top_points)} and {len(bot_points)}",
                )
                return None

            if method == "parallel":
                logger.info(f"Fitting parallel planes to {len(top_points)} + {len(bot_points)} surface points...")
                plane_normal, off1, off2 = fit_parallel_planes_from_picks(
                    top_points,
                    bot_points,
                    max_dim,
                    num_iterations,
                    learning_rate,
                )
                surface1 = evaluate_plane_on_grid_picks(plane_normal, off1, fit_resolution, max_dim)
                surface2 = evaluate_plane_on_grid_picks(plane_normal, off2, fit_resolution, max_dim)
            elif method == "coupled":
                logger.info(
                    f"Fitting coupled (shared, parallel) slab to {len(top_points)} + {len(bot_points)} "
                    f"surface points (regularization={regularization})...",
                )
                grid, o1, o2 = fit_coupled_spline_slab(
                    top_points,
                    bot_points,
                    max_dim,
                    grid_resolution,
                    num_iterations,
                    learning_rate,
                    regularization,
                )
                surface1 = evaluate_coupled_on_grid(grid, o1, fit_resolution, max_dim)
                surface2 = evaluate_coupled_on_grid(grid, o2, fit_resolution, max_dim)
            elif method == "spline":
                logger.info(f"Fitting independent splines to top/bottom surfaces (regularization={regularization})...")
                grid1 = fit_spline_surface(
                    top_points,
                    max_dim,
                    grid_resolution,
                    num_iterations,
                    learning_rate,
                    regularization,
                )
                grid2 = fit_spline_surface(
                    bot_points,
                    max_dim,
                    grid_resolution,
                    num_iterations,
                    learning_rate,
                    regularization,
                )
                surface1 = evaluate_spline_on_grid(grid1, fit_resolution, max_dim)
                surface2 = evaluate_spline_on_grid(grid2, fit_resolution, max_dim)
            else:
                logger.error(f"Unknown method '{method}' (expected spline|coupled|parallel|iou)")
                return None

        # Create closed slab mesh
        mesh = triangulate_box(surface1, surface2, fit_resolution)

        return store_mesh_with_stats(run, mesh, object_name, session_id, user_id, "slab")

    except Exception as e:
        logger.error(f"Error creating slab mesh from segmentation: {e}")
        return None


slab_from_segmentation_lazy_batch = create_lazy_batch_converter(
    converter_func=slab_from_segmentation,
    task_description="Fitting a slab to segmentations and creating slab meshes",
)
