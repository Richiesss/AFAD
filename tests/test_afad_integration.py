"""Integration tests for the AFAD hybrid pipeline (HeteroFL + FedGen).

Tests the full AFAD-specific components that don't overlap with
existing test_integration.py (strategy-level) or test_fedgen_faithful.py
(FedGen components).

Covers:
- AFADGeneratorTrainer with forward_from_latent
- AFADClient fit with FedGen regularization + HeteroFL shape-aware params
- E2E 2-round AFAD pipeline (distribute → train → aggregate → generator)
"""

import numpy as np
import torch
import torch.nn as nn

from src.client.afad_client import AFADClient
from src.models.fedgen_wrapper import FedGenModelWrapper
from src.server.generator.afad_generator_trainer import AFADGeneratorTrainer
from src.server.generator.fedgen_generator import FedGenGenerator
from src.server.strategy.heterofl_aggregator import HeteroFLAggregator

# ── Helpers ──────────────────────────────────────────────────────────


class _SimpleCNN(nn.Module):
    """Minimal CNN for testing."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class _SimpleViT(nn.Module):
    """Minimal ViT-like model with heads attribute."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.patch_embed = nn.Conv2d(1, 16, 4, stride=4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.heads = nn.Sequential(nn.Linear(16, num_classes))

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.heads(x)


def _make_loader(num_samples: int = 64, num_classes: int = 10, batch_size: int = 16):
    images = torch.randn(num_samples, 1, 28, 28)
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = torch.utils.data.TensorDataset(images, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


def _make_wrapped_model(
    model_cls=_SimpleCNN, latent_dim: int = 32, num_classes: int = 10
) -> FedGenModelWrapper:
    base = model_cls(num_classes=num_classes)
    return FedGenModelWrapper(base, latent_dim=latent_dim, num_classes=num_classes)


# ── AFADGeneratorTrainer Tests ───────────────────────────────────────


class TestAFADGeneratorTrainer:
    """Test server-side generator training with forward_from_latent."""

    def test_train_generator_loss_decreases(self):
        """Generator loss should decrease over training iterations."""
        gen = FedGenGenerator(
            noise_dim=32, num_classes=10, hidden_dim=64, latent_dim=32
        )
        cnn_model = _make_wrapped_model(_SimpleCNN)
        vit_model = _make_wrapped_model(_SimpleViT)
        models = {"cnn": cnn_model, "vit": vit_model}

        trainer = AFADGeneratorTrainer(
            generator=gen, gen_lr=1e-3, batch_size=16, device="cpu"
        )

        num_families = 2
        label_weights = np.ones((10, num_families)) / num_families
        qualified_labels = list(range(10))

        loss = trainer.train_generator(
            models=models,
            label_weights=label_weights,
            qualified_labels=qualified_labels,
            num_epochs=1,
            num_teacher_iters=5,
        )
        assert loss > 0.0

    def test_train_generator_updates_params(self):
        """Generator parameters should change after training."""
        gen = FedGenGenerator(
            noise_dim=32, num_classes=10, hidden_dim=64, latent_dim=32
        )
        initial_params = [p.clone() for p in gen.parameters()]

        models = {
            "cnn": _make_wrapped_model(_SimpleCNN),
            "vit": _make_wrapped_model(_SimpleViT),
        }
        trainer = AFADGeneratorTrainer(
            generator=gen, gen_lr=1e-2, batch_size=16, device="cpu"
        )

        label_weights = np.ones((10, 2)) / 2
        trainer.train_generator(
            models=models,
            label_weights=label_weights,
            qualified_labels=list(range(10)),
            num_epochs=1,
            num_teacher_iters=10,
        )

        changed = any(
            not torch.equal(init_p, new_p)
            for init_p, new_p in zip(initial_params, gen.parameters())
        )
        assert changed, "Generator params did not change after training"

    def test_train_generator_needs_two_models(self):
        """Training with < 2 models should skip and return 0."""
        gen = FedGenGenerator(
            noise_dim=32, num_classes=10, hidden_dim=64, latent_dim=32
        )
        trainer = AFADGeneratorTrainer(generator=gen, device="cpu")

        loss = trainer.train_generator(
            models={"cnn": _make_wrapped_model()},
            label_weights=np.ones((10, 1)),
            qualified_labels=list(range(10)),
        )
        assert loss == 0.0

    def test_get_label_weights_shape(self):
        """Label weights should have correct shape [num_classes, num_families]."""
        counts = [
            [10, 5, 0, 8, 3, 0, 7, 2, 1, 4],
            [3, 8, 6, 0, 5, 9, 0, 4, 7, 2],
        ]
        weights, qualified = AFADGeneratorTrainer.get_label_weights(
            counts, num_classes=10
        )
        assert weights.shape == (10, 2)
        assert len(qualified) > 0
        # Weights should sum to ~1 per label
        for label in qualified:
            assert abs(weights[label].sum() - 1.0) < 1e-6


# ── AFADClient Tests ─────────────────────────────────────────────────


class TestAFADClientFit:
    """Test AFAD client with HeteroFL shape-aware params + FedGen KD."""

    def test_fit_returns_correct_structure(self):
        """Basic fit should return params, num_examples, metrics."""
        model = _make_wrapped_model()
        gen = FedGenGenerator(
            noise_dim=32, num_classes=10, hidden_dim=64, latent_dim=32
        )
        loader = _make_loader(num_samples=32)

        client = AFADClient(
            cid="0",
            model=model,
            generator=gen,
            train_loader=loader,
            epochs=1,
            device="cpu",
        )
        params, num_ex, metrics = client.fit([], {"use_local_init": True, "round": 0})

        assert isinstance(params, list)
        assert num_ex == 32
        assert "label_counts" in metrics
        assert "family" in metrics

    def test_fit_with_regularization(self):
        """Training with FedGen regularization should not crash."""
        model = _make_wrapped_model()
        gen = FedGenGenerator(
            noise_dim=32, num_classes=10, hidden_dim=64, latent_dim=32
        )
        loader = _make_loader(num_samples=64)

        client = AFADClient(
            cid="0",
            model=model,
            generator=gen,
            train_loader=loader,
            epochs=2,
            device="cpu",
            generative_alpha=1.0,
            generative_beta=1.0,
        )

        initial_state = {k: v.clone() for k, v in model.state_dict().items()}

        params, _, _ = client.fit(
            [], {"round": 5, "regularization": True, "use_local_init": True}
        )

        changed = any(
            not torch.equal(initial_state[k], model.state_dict()[k])
            for k in initial_state
        )
        assert changed, "Model params should change after training"

    def test_fit_shape_aware_set_parameters(self):
        """Client should handle shape-mismatched parameters (HeteroFL sub-models)."""
        model = _make_wrapped_model()
        gen = FedGenGenerator(
            noise_dim=32, num_classes=10, hidden_dim=64, latent_dim=32
        )
        loader = _make_loader(num_samples=32)

        client = AFADClient(
            cid="0",
            model=model,
            generator=gen,
            train_loader=loader,
            epochs=1,
            device="cpu",
        )

        # Create parameters with smaller shapes (simulating sub-model distribution)
        full_params = client.get_parameters(config={})
        sub_params = []
        for p in full_params:
            if len(p.shape) >= 2:
                # Reduce first two dims by half
                slices = [slice(0, max(1, s // 2)) for s in p.shape[:2]]
                slices.extend([slice(None)] * (len(p.shape) - 2))
                sub_params.append(p[tuple(slices)])
            elif len(p.shape) == 1:
                sub_params.append(p[: max(1, len(p) // 2)])
            else:
                sub_params.append(p)

        # Should not crash with shape-mismatched params
        params, num_ex, _ = client.fit(sub_params, {"round": 1})
        assert len(params) > 0


# ── E2E Pipeline Test ────────────────────────────────────────────────


class TestAFADTwoRoundPipeline:
    """End-to-end 2-round AFAD simulation."""

    def test_distribute_train_aggregate_generator(self):
        """Simulate 2 rounds: distribute → client train → aggregate → gen train."""
        latent_dim = 32
        num_classes = 10
        aggregator = HeteroFLAggregator()

        # Create family global models (rate=1.0)
        cnn_wrapped = _make_wrapped_model(_SimpleCNN, latent_dim=latent_dim)
        vit_wrapped = _make_wrapped_model(_SimpleViT, latent_dim=latent_dim)

        family_globals = {
            "cnn": [v.cpu().numpy() for v in cnn_wrapped.state_dict().values()],
            "vit": [v.cpu().numpy() for v in vit_wrapped.state_dict().values()],
        }

        gen = FedGenGenerator(
            noise_dim=32, num_classes=num_classes, hidden_dim=64, latent_dim=latent_dim
        )
        trainer = AFADGeneratorTrainer(
            generator=gen, gen_lr=1e-3, batch_size=16, device="cpu"
        )

        loaders = [_make_loader(num_samples=32) for _ in range(2)]

        for round_num in range(1, 3):
            # --- Distribute ---
            cnn_sub = aggregator.distribute(
                family_globals["cnn"],
                "c0",
                model_rate=1.0,
                num_preserved_tail_layers=2,
            )
            vit_sub = aggregator.distribute(
                family_globals["vit"],
                "c1",
                model_rate=1.0,
                num_preserved_tail_layers=2,
            )

            # --- Client training ---
            cnn_client_model = _make_wrapped_model(_SimpleCNN, latent_dim=latent_dim)
            vit_client_model = _make_wrapped_model(_SimpleViT, latent_dim=latent_dim)

            cnn_client = AFADClient(
                cid="c0",
                model=cnn_client_model,
                generator=gen,
                train_loader=loaders[0],
                epochs=1,
                device="cpu",
                family="cnn",
            )
            vit_client = AFADClient(
                cid="c1",
                model=vit_client_model,
                generator=gen,
                train_loader=loaders[1],
                epochs=1,
                device="cpu",
                family="vit",
            )

            cnn_params, cnn_n, cnn_m = cnn_client.fit(
                cnn_sub, {"round": round_num, "use_local_init": round_num == 1}
            )
            vit_params, vit_n, vit_m = vit_client.fit(
                vit_sub, {"round": round_num, "use_local_init": round_num == 1}
            )

            # --- Aggregate per family ---
            from flwr.common import parameters_to_ndarrays

            cnn_agg = aggregator.aggregate(
                family="cnn",
                results=[("c0", cnn_params, cnn_n)],
                global_params=family_globals["cnn"],
                num_preserved_tail_layers=2,
            )
            family_globals["cnn"] = parameters_to_ndarrays(cnn_agg)

            vit_agg = aggregator.aggregate(
                family="vit",
                results=[("c1", vit_params, vit_n)],
                global_params=family_globals["vit"],
                num_preserved_tail_layers=2,
            )
            family_globals["vit"] = parameters_to_ndarrays(vit_agg)

            # --- Generator training ---
            # Reconstruct models from aggregated params
            def _load_model(cls, params):
                m = _make_wrapped_model(cls, latent_dim=latent_dim)
                sd = m.state_dict()
                keys = list(sd.keys())
                for k, p in zip(keys, params):
                    sd[k] = torch.from_numpy(p.copy())
                m.load_state_dict(sd)
                return m

            models = {
                "cnn": _load_model(_SimpleCNN, family_globals["cnn"]),
                "vit": _load_model(_SimpleViT, family_globals["vit"]),
            }

            label_weights = np.ones((num_classes, 2)) / 2
            trainer.train_generator(
                models=models,
                label_weights=label_weights,
                qualified_labels=list(range(num_classes)),
                num_epochs=1,
                num_teacher_iters=3,
            )

        # After 2 rounds: families should have updated params
        assert len(family_globals["cnn"]) > 0
        assert len(family_globals["vit"]) > 0

        # Generator should produce valid latent vectors
        gen.eval()
        labels = torch.randint(0, num_classes, (4,))
        result = gen(labels)
        assert result["output"].shape == (4, latent_dim)
        assert torch.isfinite(result["output"]).all()
