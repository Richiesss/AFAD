"""Centralized dataset configuration for AFAD experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    """Immutable dataset configuration."""

    name: str
    num_classes: int
    channels: int
    image_size: int
    mean: tuple[float, ...]
    std: tuple[float, ...]


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "mnist": DatasetConfig(
        name="mnist",
        num_classes=10,
        channels=1,
        image_size=28,
        mean=(0.1307,),
        std=(0.3081,),
    ),
    "organamnist": DatasetConfig(
        name="organamnist",
        num_classes=11,
        channels=1,
        image_size=28,
        mean=(0.4680,),
        std=(0.2974,),
    ),
}


def get_dataset_config(name: str) -> DatasetConfig:
    """Get dataset configuration by name.

    Args:
        name: Dataset name (e.g. "mnist", "organamnist")

    Returns:
        DatasetConfig for the specified dataset

    Raises:
        KeyError: If dataset name is not found
    """
    if name not in DATASET_CONFIGS:
        available = ", ".join(DATASET_CONFIGS.keys())
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    return DATASET_CONFIGS[name]
