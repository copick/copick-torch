"""
Based on https://www.kaggle.com/code/kharrington/blobdetector
"""

import csv
import os
import time

import copick
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F
import zarr
from copick.models import CopickLocation, CopickPoint
from skimage.measure import regionprops
from skimage.morphology import ball
from skimage.segmentation import watershed
from tqdm import tqdm

DEVICE = "cpu"
tomo_type = "wbp"  # See where this is used for an example of how to get the denoised
RESOLUTION_THRESHOLD = 16


def gaussian_kernel(size, sigma):
    """Generate a 3D Gaussian kernel."""
    kernel = np.fromfunction(
        lambda x, y, z: (1 / (2 * np.pi * sigma**2))
        * np.exp(
            -((x - (size[0] - 1) / 2) ** 2 + (y - (size[1] - 1) / 2) ** 2 + (z - (size[2] - 1) / 2) ** 2)
            / (2 * sigma**2),
        ),
        size,
    )
    return torch.tensor(kernel).float().unsqueeze(0).unsqueeze(0).to(DEVICE)  # Add batch and channel dimensions


def create_hessian_particle_mask(tomogram, sigma):
    """
    Generate a binary mask for dark, blob-like particles in a cryo-ET tomogram
    using Hessian-based filtering with PyTorch.

    Args:
        tomogram (torch.Tensor): The input 3D tomogram (C, D, H, W).
        sigma (float): The standard deviation for Gaussian smoothing.

    Returns:
        torch.Tensor: Binary mask highlighting dark blob-like areas in the tomogram.
    """
    kernel_size = (5, 5, 5)
    gaussian_k = gaussian_kernel(kernel_size, sigma)

    tomogram_smoothed = F.conv3d(tomogram.unsqueeze(0).unsqueeze(0), gaussian_k, padding=2).squeeze()

    # Compute Hessian components
    hessian_xx = F.conv3d(tomogram_smoothed.unsqueeze(0).unsqueeze(0), gaussian_k, padding=2)
    hessian_yy = F.conv3d(tomogram_smoothed.unsqueeze(0).unsqueeze(0), gaussian_k, padding=2)
    hessian_xy = F.conv3d(tomogram_smoothed.unsqueeze(0).unsqueeze(0), gaussian_k, padding=2)

    hessian_response = hessian_xx + hessian_yy + hessian_xy  # Simplified combination
    binary_mask = hessian_response < 0  # Adjust threshold based on your needs

    return binary_mask.squeeze().byte()


def erode_dilate_mask(mask, radius):
    """
    Perform binary erosion and dilation on a binary mask using a spherical structuring element.

    Args:
        mask (torch.Tensor): Input binary mask
        radius (int): Radius of the spherical structuring element

    Returns:
        torch.Tensor: Dilated mask after erosion and dilation operations
    """
    # Create a spherical structuring element
    radius = int(radius)  # Ensure radius is an integer
    struct_elem = ball(radius)
    struct_elem_tensor = torch.tensor(struct_elem, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)

    # Reshape mask for conv3d
    mask_reshaped = mask.unsqueeze(0).unsqueeze(0).float()  # Shape (1, 1, D, H, W)

    # Calculate padding size - ensure it's an integer
    pad_size = int(radius // 2)

    # Debug: Print shapes
    print(f"Mask shape for erosion: {mask_reshaped.shape}")
    print(f"Structuring element shape: {struct_elem_tensor.shape}")
    print(f"Padding size: {pad_size}")

    # Erosion: Use a negative structuring element for max pooling
    # Convert padding to the expected format (left, right, top, bottom, front, back)
    # Ensure all values are integers
    pad_3d = (int(pad_size), int(pad_size), int(pad_size), int(pad_size), int(pad_size), int(pad_size))

    mask_padded = F.pad(mask_reshaped, pad_3d, mode="constant", value=1)
    eroded = -F.conv3d(-mask_padded, struct_elem_tensor, stride=1, padding=0, dilation=1, groups=1)
    eroded = (eroded >= struct_elem_tensor.sum()).squeeze().byte()

    # Dilation
    mask_padded = F.pad(eroded.unsqueeze(0).unsqueeze(0).float(), pad_3d, mode="constant", value=0)
    dilated = F.conv3d(mask_padded, struct_elem_tensor, stride=1, padding=0, dilation=1, groups=1)
    dilated = (dilated > 0).squeeze().byte()

    return dilated


def distance_transform(mask):
    """
    Compute the distance transform using a simple distance transform approach.

    Args:
        mask (torch.Tensor): Binary mask tensor

    Returns:
        torch.Tensor: Distance transform result
    """
    # Ensure mask is boolean, then convert to float for distance calculation
    mask = mask.bool()
    # Invert the mask (using logical not instead of bitwise not)
    inverted_mask = (~mask).float()

    # Add batch and channel dimensions
    inverted_mask = inverted_mask.unsqueeze(0).unsqueeze(0)

    # Create kernel on the correct device
    kernel = torch.ones(1, 1, 3, 3, 3, device=DEVICE)

    # Compute distance transform using convolution
    distance = F.conv3d(inverted_mask, kernel, padding=1)

    return distance.squeeze()


def local_maxima(distance, radius):
    """
    Detect local maxima in the distance transform.

    Args:
        distance (torch.Tensor): Distance transform tensor
        radius (int): Radius for local maxima detection

    Returns:
        torch.Tensor: Binary mask of local maxima
    """
    # Ensure radius is an integer
    radius = int(radius)

    # Add batch dimension for max_pool3d
    distance = distance.unsqueeze(0)

    # Create kernel size tuple (must be odd numbers)
    kernel_size = (2 * radius + 1, 2 * radius + 1, 2 * radius + 1)

    # Compute local maxima
    maxpool = F.max_pool3d(distance, kernel_size=kernel_size, stride=1, padding=radius)

    # Compare with original distance to find local maxima
    local_max = distance == maxpool

    return local_max.squeeze()


def get_tomogram_data(run, voxel_spacing, radius):
    """
    Get tomogram data at appropriate resolution based on particle radius.

    Args:
        run: Run object
        voxel_spacing (float): Base voxel spacing
        radius (float): Particle radius

    Returns:
        tuple: (tomogram tensor, effective_voxel_spacing, scale_factor)
    """
    tomogram_wrapper = run.get_voxel_spacing(voxel_spacing).get_tomograms(
        tomo_type,
        portal_meta_query={"processing": "denoised", "processing_software": "DenoisET"},
    )[0]
    z = zarr.open(store=tomogram_wrapper.zarr(), path="/", mode="r")

    if radius <= RESOLUTION_THRESHOLD:
        # Use highest resolution
        tomogram = z["0"][:]
        effective_voxel_spacing = voxel_spacing
        scale_factor = 1
    else:
        # Use medium resolution
        tomogram = z["1"][:]
        effective_voxel_spacing = voxel_spacing * 2  # Scale factor is 2 for level 1
        scale_factor = 2

    return torch.tensor(tomogram).to(DEVICE), effective_voxel_spacing, scale_factor


def process_all_runs(root, session_id, user_id, voxel_spacing):
    """Process all runs and save results to copick."""
    results = []
    pick_id = 0

    for run in tqdm(root.runs):
        start_time = time.time()
        print(f"\nProcessing run: {run.meta.name}")

        # Process each particle type separately since they might need different resolutions
        for obj in root.pickable_objects:
            if not obj.is_particle:
                continue

            radius = obj.radius
            print(f"Processing {obj.name} with radius {radius}")

            # Get appropriate resolution data
            tomogram_tensor, effective_voxel_spacing, scale_factor = get_tomogram_data(run, voxel_spacing, radius)

            print(f"Using scale factor {scale_factor} (effective voxel spacing: {effective_voxel_spacing})")

            # Create segmentation at appropriate scale
            segmentation = create_hessian_particle_mask(tomogram_tensor, sigma=3)

            if torch.sum(segmentation) == 0:
                print(f"No particles detected in segmentation for {obj.name}")
                continue

            # Adjust radius for effective voxel spacing
            scaled_radius = radius / effective_voxel_spacing

            # Erode and dilate the segmentation
            dilated_mask = erode_dilate_mask(segmentation, scaled_radius)

            # Distance transform and local maxima detection
            distance = distance_transform(dilated_mask)
            local_max = local_maxima(distance, scaled_radius)

            # Convert tensors to numpy for watershed
            local_max_np = local_max.cpu().numpy()
            distance_np = distance.cpu().numpy()
            dilated_mask_np = dilated_mask.cpu().numpy()

            # Watershed segmentation
            markers, _ = ndi.label(local_max_np)
            watershed_labels = watershed(-distance_np, markers, mask=dilated_mask_np)

            # Extract region properties and scale coordinates back to original space
            centroids = []
            for region in regionprops(watershed_labels):
                # Scale the centroid coordinates back to original space
                centroid = np.array(region.centroid) * scale_factor
                centroids.append(centroid)  # ZYX order

            # Save centroids as picks and add to results
            if centroids:
                pick_set = run.get_picks(obj.name, user_id=user_id, session_id=session_id)
                if pick_set:
                    pick_set = pick_set[0]
                else:
                    pick_set = run.new_picks(obj.name, session_id, user_id)

                for centroid in centroids:
                    # Convert from ZYX to XYZ order and apply voxel spacing
                    x = centroid[2] * voxel_spacing  # Z -> X
                    y = centroid[1] * voxel_spacing  # Y -> Y
                    z = centroid[0] * voxel_spacing  # X -> Z

                    # Add to results list
                    row = [pick_id, run.meta.name, obj.name, x, y, z]
                    results.append(row)
                    pick_id += 1

                # Store pick set
                pick_set.points = [
                    CopickPoint(
                        location=CopickLocation(
                            x=(c[2] * voxel_spacing),
                            y=(c[1] * voxel_spacing),
                            z=(c[0] * voxel_spacing),
                        ),
                    )
                    for c in centroids
                ]
                pick_set.store()
                print(f"Saved {len(centroids)} centroids for {obj.name}")
            else:
                print(f"No valid centroids found for {obj.name}")

        # Print timing for this run
        end_time = time.time()
        print(f"Run {run.meta.name} completed in {end_time - start_time:.2f} seconds")

    print(f"\nTotal picks found: {len(results)}")


# Run the processing
# root = copick.from_file(copick_config_path)
root = copick.from_czcdp_datasets([10440], overlay_root="/tmp/test/")
results = process_all_runs(root=root, session_id="0", user_id="blobDetector", voxel_spacing=10.012)
