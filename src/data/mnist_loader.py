from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


def load_mnist_data(
    num_clients: int, batch_size: int, max_samples: int = None
) -> tuple[list[DataLoader], DataLoader]:
    """
    Load MNIST data and split into num_clients partitions.
    Returns:
        train_loaders: List of DataLoaders for each client
        test_loader: DataLoader for testing
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Download to data folder in root (likely where it expects it)
    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )

    # Subsample for debugging
    if max_samples:
        indices = list(range(min(len(train_dataset), max_samples)))
        train_dataset = Subset(train_dataset, indices)
        test_indices = list(range(min(len(test_dataset), max_samples // 5)))
        test_dataset = Subset(test_dataset, test_indices)

    # Split training set into partitions
    total_len = len(train_dataset)
    partition_size = total_len // num_clients
    lengths = [partition_size] * num_clients

    # Add remainder to last partition
    remainder = total_len - sum(lengths)
    if remainder > 0:
        lengths[-1] += remainder

    partitions = random_split(train_dataset, lengths)

    train_loaders = []
    for partition in partitions:
        train_loaders.append(DataLoader(partition, batch_size=batch_size, shuffle=True))

    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loaders, test_loader
