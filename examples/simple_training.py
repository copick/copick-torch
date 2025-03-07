import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from copick_torch import CopickDataset
import multiprocessing

def main():
    # Basic usage
    dataset = CopickDataset(
        config_path='./examples/czii_object_detection_training.json',
        boxsize=(32, 32, 32),
        augment=True,
        cache_dir='./cache',
        voxel_spacing=10.012
    )

    # Print dataset information
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.keys()}")

    # Create a weighted sampler to balance classes
    sampler = WeightedRandomSampler(
        weights=dataset.get_sample_weights(),
        num_samples=len(dataset),
        replacement=True
    )

    # Create data loader with balanced sampling
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        sampler=sampler,
        num_workers=4
    )

    # Simple training loop example
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
        torch.nn.Linear(128, len(dataset.keys()))
    )

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Print epoch statistics
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}, '
              f'Accuracy: {100.0*correct/total:.2f}%')

    print('Finished Training')

    # Get example volumes for each class
    examples, class_names = dataset.examples()
    print(f"Examples shape: {examples.shape}")
    print(f"Class names: {class_names}")

if __name__ == "__main__":
    # This is required for multiprocessing on macOS
    multiprocessing.freeze_support()
    main()