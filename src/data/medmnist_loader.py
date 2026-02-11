"""OrganAMNIST data loader with Dirichlet Non-IID partitioning."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset_config import get_dataset_config


def _medmnist_to_tensors(
    medmnist_dataset, transform
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a medmnist dataset to pre-transformed (images, labels) tensors.

    This avoids custom Dataset classes that fail Ray pickle deserialization
    in Flower simulation workers (workers can't import src.data).
    """
    images = []
    labels = []
    for i in range(len(medmnist_dataset)):
        img, label = medmnist_dataset[i]
        img = transform(img)
        images.append(img)
        # medmnist returns label as ndarray shape (1,) -> scalar int
        if hasattr(label, "__len__"):
            labels.append(int(label[0]))
        else:
            labels.append(int(label))
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def _dirichlet_partition(
    targets: np.ndarray,
    num_clients: int,
    alpha: float,
    num_classes: int,
    seed: int = 42,
) -> list[np.ndarray]:
    """Partition dataset indices using Dirichlet distribution.

    For each class, samples a proportion vector from Dir(alpha) and assigns
    indices to clients proportionally.

    Args:
        targets: Array of integer labels (N,)
        num_clients: Number of clients to partition to
        alpha: Dirichlet concentration parameter (lower = more heterogeneous)
        num_classes: Number of classes in the dataset
        seed: Random seed for reproducibility

    Returns:
        List of index arrays, one per client
    """
    rng = np.random.default_rng(seed)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for cls in range(num_classes):
        cls_indices = np.where(targets == cls)[0]
        rng.shuffle(cls_indices)

        # Sample proportions from Dirichlet
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        # Convert proportions to counts
        counts = (proportions * len(cls_indices)).astype(int)
        # Distribute remainder to largest proportions
        remainder = len(cls_indices) - counts.sum()
        for i in range(remainder):
            counts[np.argmax(proportions - counts / max(1, len(cls_indices)))] += 1

        # Assign indices
        start = 0
        for client_id in range(num_clients):
            end = start + counts[client_id]
            client_indices[client_id].extend(cls_indices[start:end].tolist())
            start = end

    return [np.array(indices) for indices in client_indices]


def load_organamnist_data(
    num_clients: int,
    batch_size: int,
    alpha: float = 0.5,
    distribution: str = "non_iid",
    seed: int = 42,
    max_samples: int | None = None,
) -> tuple[list[DataLoader], DataLoader]:
    """Load OrganAMNIST and partition across clients.

    Args:
        num_clients: Number of client partitions
        batch_size: Batch size for DataLoaders
        alpha: Dirichlet concentration (only used if distribution="non_iid")
        distribution: "iid" or "non_iid"
        seed: Random seed
        max_samples: Limit training samples (for testing/debugging)

    Returns:
        (train_loaders, test_loader) matching load_mnist_data interface
    """
    from medmnist import OrganAMNIST
    from torchvision import transforms

    cfg = get_dataset_config("organamnist")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std),
        ]
    )

    train_raw = OrganAMNIST(split="train", download=True, root="./data")
    test_raw = OrganAMNIST(split="test", download=True, root="./data")

    # Pre-convert to tensors so DataLoaders only contain standard PyTorch
    # types (TensorDataset) that Ray workers can deserialize without
    # importing src.data.
    train_images, train_labels = _medmnist_to_tensors(train_raw, transform)
    test_images, test_labels = _medmnist_to_tensors(test_raw, transform)

    targets = train_labels.numpy()

    if max_samples is not None:
        train_images = train_images[:max_samples]
        train_labels = train_labels[:max_samples]
        targets = targets[:max_samples]

    if distribution == "iid":
        rng = np.random.default_rng(seed)
        all_indices = np.arange(len(targets))
        rng.shuffle(all_indices)
        splits = np.array_split(all_indices, num_clients)
    else:
        splits = _dirichlet_partition(
            targets, num_clients, alpha, cfg.num_classes, seed
        )

    train_loaders = []
    for indices in splits:
        idx = indices.tolist()
        client_dataset = TensorDataset(train_images[idx], train_labels[idx])
        train_loaders.append(
            DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
        )

    test_loader = DataLoader(
        TensorDataset(test_images, test_labels), batch_size=batch_size
    )

    return train_loaders, test_loader
