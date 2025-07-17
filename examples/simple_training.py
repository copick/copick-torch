import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from copick_torch import CopickDataset


def main():
    # Create cache directory if it doesn't exist
    os.makedirs("./cache", exist_ok=True)

    # Basic usage with background sampling and parquet caching
    dataset = CopickDataset(
        config_path="./examples/czii_object_detection_training.json",
        boxsize=(32, 32, 32),
        augment=True,
        cache_dir="./cache",
        cache_format="parquet",  # Use parquet caching instead of pickle
        voxel_spacing=10.012,
        include_background=True,  # Enable background sampling
        background_ratio=0.2,  # 20% of particles will be background samples
        min_background_distance=48,  # Min distance in voxels from particles
    )

    # Print dataset information
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.keys()}")

    # Show class distribution
    distribution = dataset.get_class_distribution()
    print("\nClass Distribution:")
    for class_name, count in distribution.items():
        print(f"  {class_name}: {count} samples")

    # Create a weighted sampler to balance classes (including background)
    sampler = WeightedRandomSampler(weights=dataset.get_sample_weights(), num_samples=len(dataset), replacement=True)

    # Create data loader with balanced sampling
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler, num_workers=4)

    # Simple training loop example
    # Expand model to handle background class (labeled as -1)
    num_classes = len(dataset.keys()) + 1  # Add one for background
    model = torch.nn.Sequential(
        torch.nn.Conv3d(1, 16, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool3d(2),
        torch.nn.Conv3d(16, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool3d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(32 * 8 * 8 * 8, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, num_classes),
    )

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #         'mps' if torch.backends.mps.is_available() else
    # mps still doesnt have max pool 3d

    model = model.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            # Shift labels to handle -1 for background
            # Convert:
            #   -1 (background) → 0
            #   0, 1, 2, ... (molecules) → 1, 2, 3, ...
            shifted_labels = labels + 1

            inputs, shifted_labels = inputs.to(device), shifted_labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, shifted_labels)
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += shifted_labels.size(0)
            correct += predicted.eq(shifted_labels).sum().item()

        # Print epoch statistics
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}, "
            f"Accuracy: {100.0*correct/total:.2f}%",
        )

    print("Finished Training")

    # Get example volumes for each class
    examples, class_names = dataset.examples()
    print(f"Examples shape: {examples.shape}")
    print(f"Class names: {class_names}")


def visualize_examples(dataset):
    """Visualize example volumes from each class."""
    examples, class_names = dataset.examples()

    if examples is None:
        print("No examples available")
        return

    # Convert to numpy for visualization
    examples = examples.numpy()

    # Create a figure with subplots
    n_examples = len(examples)
    fig, axes = plt.subplots(1, n_examples, figsize=(4 * n_examples, 4))

    # Handle case with only one example
    if n_examples == 1:
        axes = [axes]

    # Plot middle slice of each example
    for i, (volume, class_name) in enumerate(zip(examples, class_names)):
        # Get middle slice (channel, z, y, x)
        middle_slice = volume[0, volume.shape[1] // 2, :, :]

        # Display the slice
        im = axes[i].imshow(middle_slice, cmap="gray")
        axes[i].set_title(f"Class: {class_name}")
        axes[i].axis("off")
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("class_examples.png")
    plt.show()


if __name__ == "__main__":
    # This is required for multiprocessing on macOS
    multiprocessing.freeze_support()
    main()

    # Create a small dataset for visualization
    vis_dataset = CopickDataset(
        config_path="./examples/czii_object_detection_training.json",
        boxsize=(32, 32, 32),
        augment=False,
        cache_dir="./cache",
        include_background=True,
        max_samples=100,  # Limit samples for faster loading
    )

    # Visualize examples
    visualize_examples(vis_dataset)
