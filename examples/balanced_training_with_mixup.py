"""
Simple training example using class-balanced sampling and mixup augmentation.

This script demonstrates how to use the SimpleCopickDataset with class balancing and mixup augmentation.
"""

import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from copick_torch import ClassBalancedSampler, MixupAugmentation, SimpleCopickDataset, setup_logging


def main():
    # Set up logging
    setup_logging()

    # Create cache directory if it doesn't exist
    os.makedirs("./cache", exist_ok=True)

    # Basic usage with background sampling and parquet caching
    dataset = SimpleCopickDataset(
        config_path="./examples/czii_object_detection_training.json",
        boxsize=(32, 32, 32),
        augment=True,  # Enable basic augmentations in the dataset
        cache_dir="./cache",
        cache_format="parquet",
        voxel_spacing=10.012,
        include_background=True,
        background_ratio=0.2,
        min_background_distance=48,
    )

    # Print dataset information
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.keys()}")

    # Show class distribution
    distribution = dataset.get_class_distribution()
    print("\nClass Distribution:")
    for class_name, count in distribution.items():
        print(f"  {class_name}: {count} samples")

    # Create a class-balanced sampler
    # Get all labels for the sampler
    labels = [dataset[i][1] for i in range(len(dataset))]
    sampler = ClassBalancedSampler(labels=labels, num_samples=len(dataset), replacement=True)

    # Create data loader with balanced sampling
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4)

    # Create mixup augmentation
    mixup = MixupAugmentation(alpha=0.2)

    # Simple training loop example
    # Expand model to handle background class (labeled as -1)
    num_classes = len(dataset.keys()) + (1 if "background" in distribution else 0)
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
            # Ensure labels are non-negative for CrossEntropyLoss
            # Convert -1 (background) to num_classes-1
            labels = labels.clone()
            labels[labels == -1] = num_classes - 1

            # Apply mixup augmentation
            inputs, targets_a, targets_b, lam = mixup(inputs, labels)

            # Move data to device
            inputs = inputs.to(device)
            targets_a = targets_a.to(device)
            targets_b = targets_b.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss with mixup
            loss = mixup.mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update statistics
            running_loss += loss.item()

            # For accuracy calculation with mixup, use the dominant label
            _, predicted = outputs.max(1)
            batch_size = targets_a.size(0)
            total += batch_size

            # Calculate accuracy based on the dominant mixup component
            correct_a = predicted.eq(targets_a).float()
            correct_b = predicted.eq(targets_b).float()
            correct_mixed = lam * correct_a + (1 - lam) * correct_b
            correct += correct_mixed.sum().item()

        # Print epoch statistics
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}, "
            f"Accuracy: {100.0*correct/total:.2f}%",
        )

    print("Finished Training")

    # Save the model
    torch.save(model.state_dict(), "copick_model.pth")
    print("Model saved to copick_model.pth")


def visualize_batch_with_mixup():
    """Visualize a batch with mixup applied."""
    # Create a small dataset for visualization
    dataset = SimpleCopickDataset(
        config_path="./examples/czii_object_detection_training.json",
        boxsize=(32, 32, 32),
        augment=False,
        cache_dir="./cache",
        include_background=True,
        max_samples=100,
    )

    # Create a dataloader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Get a batch
    inputs, labels = next(iter(dataloader))

    # Create mixup augmentation
    mixup = MixupAugmentation(alpha=0.5)

    # Apply mixup
    mixed_inputs, labels_a, labels_b, lam = mixup(inputs, labels)

    # Convert labels to class names
    class_names = dataset.keys() + ["background"]
    labels_a_names = [class_names[l] if l >= 0 else "background" for l in labels_a.numpy()]
    labels_b_names = [class_names[l] if l >= 0 else "background" for l in labels_b.numpy()]

    # Visualize original and mixed samples
    fig, axes = plt.subplots(2, 8, figsize=(20, 6))

    # Show original inputs
    for i in range(8):
        # Get middle slice of each volume
        middle_slice = inputs[i, 0, inputs.shape[2] // 2, :, :]
        axes[0, i].imshow(middle_slice.numpy(), cmap="gray")
        axes[0, i].set_title(f"Original: {labels_a_names[i]}")
        axes[0, i].axis("off")

    # Show mixed inputs
    for i in range(8):
        # Get middle slice of each volume
        middle_slice = mixed_inputs[i, 0, mixed_inputs.shape[2] // 2, :, :]
        axes[1, i].imshow(middle_slice.numpy(), cmap="gray")
        axes[1, i].set_title(f"Mix: {labels_a_names[i]} ({lam:.2f})\n+ {labels_b_names[i]} ({1-lam:.2f})")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("mixup_visualization.png")
    plt.show()

    print("Mixup visualization saved to mixup_visualization.png")


if __name__ == "__main__":
    # This is required for multiprocessing on macOS
    multiprocessing.freeze_support()

    # Uncomment to run training
    main()

    # Uncomment to visualize mixup
    # visualize_batch_with_mixup()
