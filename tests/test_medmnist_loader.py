"""Tests for MedMNIST loader and Dirichlet partitioning."""

import numpy as np
import pytest

from src.data.medmnist_loader import _dirichlet_partition


class TestDirichletPartition:
    """Test _dirichlet_partition as a pure function."""

    def _make_targets(self, num_classes: int = 10, per_class: int = 100) -> np.ndarray:
        """Create balanced targets array."""
        return np.repeat(np.arange(num_classes), per_class)

    def test_all_indices_assigned(self):
        """Every index must appear exactly once across all clients."""
        targets = self._make_targets(num_classes=10, per_class=100)
        splits = _dirichlet_partition(targets, num_clients=5, alpha=0.5, num_classes=10)
        all_indices = np.concatenate(splits)
        assert len(all_indices) == len(targets)
        assert set(all_indices) == set(range(len(targets)))

    def test_num_clients_correct(self):
        targets = self._make_targets()
        splits = _dirichlet_partition(
            targets, num_clients=10, alpha=0.5, num_classes=10
        )
        assert len(splits) == 10

    def test_seed_reproducibility(self):
        targets = self._make_targets()
        splits1 = _dirichlet_partition(
            targets, num_clients=5, alpha=0.5, num_classes=10, seed=42
        )
        splits2 = _dirichlet_partition(
            targets, num_clients=5, alpha=0.5, num_classes=10, seed=42
        )
        for s1, s2 in zip(splits1, splits2):
            np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_produce_different_splits(self):
        targets = self._make_targets()
        splits1 = _dirichlet_partition(
            targets, num_clients=5, alpha=0.5, num_classes=10, seed=42
        )
        splits2 = _dirichlet_partition(
            targets, num_clients=5, alpha=0.5, num_classes=10, seed=99
        )
        # At least one client should have different indices
        any_different = any(
            not np.array_equal(s1, s2) for s1, s2 in zip(splits1, splits2)
        )
        assert any_different

    def test_high_alpha_produces_balanced_splits(self):
        """High alpha (100) should produce roughly equal-sized partitions."""
        targets = self._make_targets(num_classes=10, per_class=1000)
        splits = _dirichlet_partition(
            targets, num_clients=5, alpha=100.0, num_classes=10
        )
        sizes = [len(s) for s in splits]
        expected = len(targets) / 5
        for s in sizes:
            assert abs(s - expected) / expected < 0.15  # Within 15%

    def test_low_alpha_produces_unbalanced_splits(self):
        """Low alpha (0.01) should produce very unequal partitions."""
        targets = self._make_targets(num_classes=10, per_class=1000)
        splits = _dirichlet_partition(
            targets, num_clients=5, alpha=0.01, num_classes=10
        )
        sizes = [len(s) for s in splits]
        # Coefficient of variation should be high
        cv = np.std(sizes) / np.mean(sizes)
        assert cv > 0.1  # Some significant variation

    def test_11_classes_organamnist(self):
        """Verify correct behavior with 11 classes (OrganAMNIST)."""
        targets = self._make_targets(num_classes=11, per_class=100)
        splits = _dirichlet_partition(
            targets, num_clients=10, alpha=0.5, num_classes=11
        )
        assert len(splits) == 10
        all_indices = np.concatenate(splits)
        assert len(all_indices) == 1100
        assert set(all_indices) == set(range(1100))

    def test_no_empty_clients_moderate_alpha(self):
        """With moderate alpha, no client should be completely empty."""
        targets = self._make_targets(num_classes=10, per_class=100)
        splits = _dirichlet_partition(targets, num_clients=5, alpha=0.5, num_classes=10)
        for i, s in enumerate(splits):
            assert len(s) > 0, f"Client {i} has no data"


class TestLoadOrganAMNIST:
    """Integration tests for load_organamnist_data (requires medmnist download)."""

    @pytest.fixture(autouse=True)
    def _check_medmnist(self):
        """Skip if medmnist data is not available."""
        try:
            from medmnist import OrganAMNIST

            OrganAMNIST(split="train", download=True, root="./data")
        except Exception:
            pytest.skip("medmnist data not available")

    def test_load_iid_returns_correct_structure(self):
        from src.data.medmnist_loader import load_organamnist_data

        train_loaders, test_loader = load_organamnist_data(
            num_clients=3, batch_size=32, distribution="iid"
        )
        assert len(train_loaders) == 3
        assert test_loader is not None

    def test_load_non_iid_returns_correct_structure(self):
        from src.data.medmnist_loader import load_organamnist_data

        train_loaders, test_loader = load_organamnist_data(
            num_clients=5, batch_size=32, alpha=0.5, distribution="non_iid"
        )
        assert len(train_loaders) == 5

    def test_labels_are_integers(self):
        from src.data.medmnist_loader import load_organamnist_data

        train_loaders, _ = load_organamnist_data(
            num_clients=2, batch_size=16, distribution="iid"
        )
        images, labels = next(iter(train_loaders[0]))
        assert labels.dtype in (
            torch.int64,
            torch.int32,
            torch.long,
        ), f"Expected int labels, got {labels.dtype}"
        assert images.shape[1] == 1  # 1 channel
        assert images.shape[2] == 28


# Only needed for the labels dtype check
import torch  # noqa: E402
