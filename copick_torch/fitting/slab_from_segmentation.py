"""Fit parallel planes to a segmentation volume and create a closed slab mesh."""

from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import tqdm
from copick.util.log import get_logger
from copick_utils.converters.converter_common import store_mesh_with_stats
from copick_utils.converters.lazy_converter import create_lazy_batch_converter
from copick_utils.converters.slab_common import triangulate_box
from skimage import measure

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
    fit_resolution: Tuple[int, int] = (50, 50),
    num_iterations: int = 20,
    learning_rate: float = 0.1,
    **kwargs,
) -> Optional[Tuple["CopickMesh", Dict[str, int]]]:
    """Create a closed slab mesh by fitting parallel planes to a segmentation.

    Args:
        segmentation: CopickSegmentation object.
        run: CopickRun object.
        object_name: Name for the output mesh object.
        session_id: Session ID for the output mesh.
        user_id: User ID for the output mesh.
        label: Label index to extract from the segmentation.
        fit_resolution: Output mesh grid resolution (rows, cols).
        num_iterations: Number of optimizer iterations.
        learning_rate: Learning rate for Adam optimizer.

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

        # Fit parallel planes
        logger.info("Fitting parallel planes to segmentation...")
        plane_normal, top_offset, bot_offset = fit_parallel_planes(binary, num_iterations, learning_rate)

        # Compute physical dimensions
        voxel_size = segmentation.voxel_size
        max_dim = [d * voxel_size for d in vol.shape[::-1]]  # x, y, z

        # Evaluate planes on grid
        top_points = evaluate_plane_on_grid(plane_normal, top_offset, fit_resolution, max_dim)
        bot_points = evaluate_plane_on_grid(plane_normal, bot_offset, fit_resolution, max_dim)

        # Create closed slab mesh
        mesh = triangulate_box(bot_points, top_points, fit_resolution)

        return store_mesh_with_stats(run, mesh, object_name, session_id, user_id, "slab")

    except Exception as e:
        logger.error(f"Error creating slab mesh from segmentation: {e}")
        return None


slab_from_segmentation_lazy_batch = create_lazy_batch_converter(
    converter_func=slab_from_segmentation,
    task_description="Fitting parallel planes to segmentations and creating slab meshes",
)
