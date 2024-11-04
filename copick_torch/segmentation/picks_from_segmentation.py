import torch
import numpy as np
from scipy.ndimage import binary_dilation
from tqdm import tqdm
import numpy as np
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.measure import regionprops
from skimage.morphology import binary_erosion, binary_dilation, ball


def extract_coords_torch(labelmap, label, pickable_object, voxel_size, min_protein_size, remove_index=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labelmap = torch.tensor(labelmap, device=device, dtype=torch.float32)

    # Filter for the desired label
    mask = (labelmap == label).float()

    # Morphological dilation to identify clusters
    structure = torch.ones((3, 3, 3), device=device)
    dilated_mask = torch.from_numpy(binary_dilation(mask.cpu().numpy(), structure.cpu().numpy())).to(device)

    # Approximate center of mass
    mask_coords = dilated_mask.nonzero(as_tuple=False).float()
    if mask_coords.numel() == 0:
        return torch.zeros((0, 6), device=device)

    min_object_size = (4 / 3) * np.pi * ((pickable_object.radius / voxel_size) ** 3) * min_protein_size
    batch_size = 1024
    com_list = []
    
    for i in range(0, mask_coords.shape[0], batch_size):
        batch = mask_coords[i:i + batch_size]
        com = batch.mean(dim=0)
        if (batch.shape[0] * voxel_size**3) >= min_object_size:
            com_list.append(com)

    deepFinderCoords = torch.stack(com_list, dim=0)
    if remove_index < deepFinderCoords.shape[0]:
        deepFinderCoords = torch.cat([deepFinderCoords[:remove_index], deepFinderCoords[remove_index+1:]], dim=0)

    threshold = pickable_object.radius / (voxel_size * 3)
    deepFinderCoords = remove_duplicates_gpu(deepFinderCoords, threshold)

    deepFinderCoords = torch.cat([deepFinderCoords, torch.zeros_like(deepFinderCoords)], dim=1)
    deepFinderCoords *= voxel_size

    return deepFinderCoords


def remove_duplicates_gpu(coords, threshold):
    if coords.shape[0] == 0:
        return coords
    
    dists = torch.cdist(coords, coords)
    mask = torch.triu(dists < threshold, diagonal=1)
    unique_mask = (~mask.any(dim=0)).nonzero(as_tuple=False).squeeze()
    return coords[unique_mask]


def picks_from_segmentation(segmentation, segmentation_idx, maxima_filter_size, min_particle_size, max_particle_size, session_id, user_id, pickable_object, run, voxel_spacing=1):
    """
    Process a specific label in the segmentation, extract centroids, and save them as picks.

    Args:
        segmentation (np.ndarray): Multilabel segmentation array.
        segmentation_idx (int): The specific label from the segmentation to process.
        maxima_filter_size (int): Size of the maximum detection filter.
        min_particle_size (int): Minimum size threshold for particles.
        max_particle_size (int): Maximum size threshold for particles.
        session_id (str): Session ID for pick saving.
        user_id (str): User ID for pick saving.
        pickable_object (str): The name of the object to save picks for.
        run: A Copick run object that manages pick saving.
        voxel_spacing (int): The voxel spacing used to scale pick locations (default 1).
    """
    # Create a binary mask for the specific segmentation label
    binary_mask = (segmentation == segmentation_idx).astype(int)

    # Skip if the segmentation label is not present
    if np.sum(binary_mask) == 0:
        print(f"No segmentation with label {segmentation_idx} found.")
        return

    # Structuring element for erosion and dilation
    struct_elem = ball(1)
    eroded = binary_erosion(binary_mask, struct_elem)
    dilated = binary_dilation(eroded, struct_elem)

    # Distance transform and local maxima detection
    distance = ndi.distance_transform_edt(dilated)
    local_max = (distance == ndi.maximum_filter(distance, footprint=np.ones((maxima_filter_size,)*3)))

    # Watershed segmentation
    markers, _ = ndi.label(local_max)
    watershed_labels = watershed(-distance, markers, mask=dilated)

    # Extract region properties and filter based on particle size
    all_centroids = []
    for region in regionprops(watershed_labels):
        if min_particle_size <= region.area <= max_particle_size:
            all_centroids.append(region.centroid)

    # Convert centroids to GPU tensors and filter
    if all_centroids:
        centroids_np = np.array(all_centroids)
        deepFinderCoords = extract_coords_torch(centroids_np, segmentation_idx, pickable_object, voxel_spacing, min_particle_size)

        threshold = np.ceil(  pickable_object.radius / (voxel_size * 3) )

        try: 
            # Remove Double Counted Coordinates
            deepFinderCoords = remove_duplicates_gpu(deepFinderCoords, threshold)

            # Append Euler Angles to Coordinates [ Expand Dimensions from Nx3 -> Nx6 ]
            deepFinderCoords = np.concatenate((deepFinderCoords, np.zeros(deepFinderCoords.shape)),axis=1)

            # Convert from Voxel to Physical Units
            deepFinderCoords *= voxel_size

        except Exception as e:
            print(f"Error processing label {label} in tomo {copick_run}: {e}")
            deepFinderCoords = np.array([]).reshape(0,6)

        # Save centroids as picks
        pick_set = run.new_picks(pickable_object, session_id, user_id)
        pick_set.points = [{'x': c[2].item(), 'y': c[1].item(), 'z': c[0].item()} for c in deepFinderCoords.cpu().numpy()]
        pick_set.store()
        print(f"Centroids for label {segmentation_idx} saved successfully.")
        return pick_set
    else:
        print(f"No valid centroids found for label {segmentation_idx}.")
        return None

