"""Tests for FedProx proximal regularization in AFADClient."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.client.afad_client import AFADClient


def _make_simple_model() -> nn.Module:
    """Simple model for fast testing."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )


def _make_loader(n_samples: int = 64, num_classes: int = 10) -> DataLoader:
    """Create a small synthetic DataLoader."""
    images = torch.randn(n_samples, 1, 28, 28)
    labels = torch.randint(0, num_classes, (n_samples,))
    return DataLoader(TensorDataset(images, labels), batch_size=16)


class TestFedProxRegularization:
    """Verify FedProx proximal term behavior."""

    def test_mu_zero_same_as_standard_sgd(self):
        """With mu=0, FedProx should behave identically to standard training."""
        torch.manual_seed(0)
        model_a = _make_simple_model()
        loader = _make_loader()

        # Train with mu=0 (FedProx disabled)
        client_a = AFADClient(
            cid="0",
            model=model_a,
            train_loader=loader,
            epochs=2,
            device="cpu",
            num_classes=10,
        )
        config_a = {"fedprox_mu": 0.0}
        params_a, _, _ = client_a.fit([], config_a)

        # Train with default (no fedprox_mu in config)
        torch.manual_seed(0)
        model_b = _make_simple_model()
        client_b = AFADClient(
            cid="1",
            model=model_b,
            train_loader=loader,
            epochs=2,
            device="cpu",
            num_classes=10,
        )
        config_b = {}
        params_b, _, _ = client_b.fit([], config_b)

        # Should produce identical results
        for pa, pb in zip(params_a, params_b):
            assert torch.allclose(torch.tensor(pa), torch.tensor(pb), atol=1e-6), (
                "mu=0 should match standard SGD"
            )

    def test_mu_positive_stays_closer_to_global(self):
        """With mu>0, final params should be closer to global than without."""
        loader = _make_loader(n_samples=128)

        # Global model (initial params before local training)
        torch.manual_seed(42)
        global_model = _make_simple_model()
        global_params = [p.clone().detach() for p in global_model.parameters()]
        global_ndarrays = [p.numpy() for p in global_params]

        # Train WITHOUT FedProx
        torch.manual_seed(42)
        model_no_prox = _make_simple_model()
        client_no_prox = AFADClient(
            cid="0",
            model=model_no_prox,
            train_loader=loader,
            epochs=3,
            device="cpu",
            num_classes=10,
        )
        params_no_prox, _, _ = client_no_prox.fit(global_ndarrays, {"fedprox_mu": 0.0})

        # Train WITH FedProx (mu=1.0, strong regularization for clear effect)
        torch.manual_seed(42)
        model_prox = _make_simple_model()
        client_prox = AFADClient(
            cid="1",
            model=model_prox,
            train_loader=loader,
            epochs=3,
            device="cpu",
            num_classes=10,
        )
        params_prox, _, _ = client_prox.fit(global_ndarrays, {"fedprox_mu": 1.0})

        # Compute L2 distance from global for each
        dist_no_prox = sum(
            ((torch.tensor(p) - g) ** 2).sum().item()
            for p, g in zip(params_no_prox, global_params)
        )
        dist_prox = sum(
            ((torch.tensor(p) - g) ** 2).sum().item()
            for p, g in zip(params_prox, global_params)
        )

        assert dist_prox < dist_no_prox, (
            f"FedProx (dist={dist_prox:.4f}) should stay closer to global "
            f"than standard SGD (dist={dist_no_prox:.4f})"
        )

    def test_mu_from_config_propagation(self):
        """Verify fedprox_mu is read from fit config dict."""
        loader = _make_loader(n_samples=32)
        model = _make_simple_model()
        global_ndarrays = [p.detach().numpy() for p in model.parameters()]

        client = AFADClient(
            cid="0",
            model=model,
            train_loader=loader,
            epochs=1,
            device="cpu",
            num_classes=10,
        )

        # Should not raise with mu in config
        params, num_examples, metrics = client.fit(
            global_ndarrays, {"fedprox_mu": 0.05}
        )
        assert len(params) > 0
        assert num_examples == 32

    def test_stronger_mu_produces_smaller_drift(self):
        """Higher mu should result in smaller parameter drift from global."""
        loader = _make_loader(n_samples=128)

        dists = {}
        for mu in [0.0, 0.1, 1.0]:
            torch.manual_seed(42)
            model = _make_simple_model()
            global_params = [p.clone().detach() for p in model.parameters()]
            global_ndarrays = [p.numpy() for p in global_params]

            client = AFADClient(
                cid="0",
                model=model,
                train_loader=loader,
                epochs=3,
                device="cpu",
                num_classes=10,
            )
            params, _, _ = client.fit(global_ndarrays, {"fedprox_mu": mu})

            dist = sum(
                ((torch.tensor(p) - g) ** 2).sum().item()
                for p, g in zip(params, global_params)
            )
            dists[mu] = dist

        # Monotonically decreasing drift with increasing mu
        assert dists[0.0] > dists[0.1], (
            f"mu=0.0 drift ({dists[0.0]:.4f}) should > mu=0.1 ({dists[0.1]:.4f})"
        )
        assert dists[0.1] > dists[1.0], (
            f"mu=0.1 drift ({dists[0.1]:.4f}) should > mu=1.0 ({dists[1.0]:.4f})"
        )
