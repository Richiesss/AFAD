"""Tests for dataset configuration."""

import pytest

from src.data.dataset_config import DATASET_CONFIGS, get_dataset_config


class TestDatasetConfig:
    def test_mnist_config_exists(self):
        cfg = get_dataset_config("mnist")
        assert cfg.name == "mnist"
        assert cfg.num_classes == 10
        assert cfg.channels == 1
        assert cfg.image_size == 28

    def test_organamnist_config_exists(self):
        cfg = get_dataset_config("organamnist")
        assert cfg.name == "organamnist"
        assert cfg.num_classes == 11
        assert cfg.channels == 1
        assert cfg.image_size == 28

    def test_unknown_dataset_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown dataset"):
            get_dataset_config("nonexistent")

    def test_mean_std_are_tuples(self):
        for name, cfg in DATASET_CONFIGS.items():
            assert isinstance(cfg.mean, tuple), f"{name}.mean is not tuple"
            assert isinstance(cfg.std, tuple), f"{name}.std is not tuple"
            assert len(cfg.mean) == cfg.channels
            assert len(cfg.std) == cfg.channels

    def test_config_is_frozen(self):
        cfg = get_dataset_config("mnist")
        with pytest.raises(AttributeError):
            cfg.num_classes = 99

    def test_organamnist_mean_std_reasonable(self):
        cfg = get_dataset_config("organamnist")
        assert 0.0 < cfg.mean[0] < 1.0
        assert 0.0 < cfg.std[0] < 1.0

    def test_all_configs_have_positive_classes(self):
        for name, cfg in DATASET_CONFIGS.items():
            assert cfg.num_classes > 0, f"{name} has non-positive num_classes"
