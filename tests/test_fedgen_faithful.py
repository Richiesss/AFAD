"""Tests for faithful FedGen implementation (Zhu et al., ICML 2021).

Covers:
- FedGenGenerator: latent-space output, diversity loss
- FedGenModelWrapper: forward, forward_from_latent, classifier detection
- FedGenClient: client-side KD with α/β decay
- Scaler: training-only behavior
"""

import torch
import torch.nn as nn

from src.client.fedgen_client import (
    DECAY_RATE,
    DEFAULT_GENERATIVE_ALPHA,
    FedGenClient,
    exp_lr_decay,
)
from src.models.fedgen_wrapper import FedGenModelWrapper, _find_classifier_info
from src.models.scaler import Scaler
from src.server.generator.fedgen_generator import FedGenGenerator

# ── Helpers ──────────────────────────────────────────────────────────


class _SimpleCNN(nn.Module):
    """Minimal CNN for testing wrapper detection."""

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
    """Create a simple DataLoader for testing."""
    images = torch.randn(num_samples, 1, 28, 28)
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = torch.utils.data.TensorDataset(images, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


# ── Scaler Tests ─────────────────────────────────────────────────────


class TestScalerTrainingOnly:
    """Verify Scaler matches original paper: scale during training, identity during eval."""

    def test_scaler_divides_during_training(self):
        scaler = Scaler(rate=0.5)
        scaler.train()
        x = torch.ones(2, 4)
        result = scaler(x)
        assert torch.allclose(result, x / 0.5)

    def test_scaler_identity_during_eval(self):
        scaler = Scaler(rate=0.5)
        scaler.eval()
        x = torch.ones(2, 4)
        result = scaler(x)
        assert torch.allclose(result, x)

    def test_scaler_rate_1_is_identity_always(self):
        scaler = Scaler(rate=1.0)
        x = torch.randn(2, 4)
        scaler.train()
        assert torch.allclose(scaler(x), x)
        scaler.eval()
        assert torch.allclose(scaler(x), x)


# ── FedGenGenerator Tests ───────────────────────────────────────────


class TestFedGenGenerator:
    def setup_method(self):
        self.gen = FedGenGenerator(
            noise_dim=32, num_classes=10, hidden_dim=256, latent_dim=32
        )

    def test_output_is_latent_not_image(self):
        """Generator must output flat latent vectors, not images."""
        labels = torch.randint(0, 10, (4,))
        result = self.gen(labels)
        assert result["output"].shape == (4, 32)
        assert result["output"].ndim == 2  # NOT 4D (images would be 4D)

    def test_output_shape_matches_latent_dim(self):
        gen = FedGenGenerator(latent_dim=64)
        labels = torch.randint(0, 10, (8,))
        result = gen(labels)
        assert result["output"].shape == (8, 64)

    def test_eps_shape(self):
        labels = torch.randint(0, 10, (4,))
        result = self.gen(labels)
        assert result["eps"].shape == (4, 32)

    def test_different_labels_produce_different_outputs(self):
        """Conditioning on different labels should produce different latents."""
        self.gen.eval()
        torch.manual_seed(42)
        labels_a = torch.zeros(4, dtype=torch.long)
        result_a = self.gen(labels_a)
        torch.manual_seed(42)
        labels_b = torch.ones(4, dtype=torch.long) * 5
        result_b = self.gen(labels_b)
        assert not torch.allclose(result_a["output"], result_b["output"])

    def test_diversity_loss_returns_scalar(self):
        labels = torch.randint(0, 10, (8,))
        result = self.gen(labels)
        loss = self.gen.diversity_loss(result["eps"], result["output"])
        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_diversity_loss_small_batch(self):
        labels = torch.randint(0, 10, (2,))
        result = self.gen(labels)
        loss = self.gen.diversity_loss(result["eps"], result["output"])
        assert loss.shape == ()

    def test_crossentropy_loss_per_sample(self):
        logp = torch.randn(4, 10).log_softmax(dim=1)
        labels = torch.randint(0, 10, (4,))
        losses = FedGenGenerator.crossentropy_loss(logp, labels)
        assert losses.shape == (4,)
        assert torch.all(losses >= 0.0)

    def test_eval_mode_batch_one(self):
        self.gen.eval()
        labels = torch.randint(0, 10, (1,))
        result = self.gen(labels)
        assert result["output"].shape == (1, 32)

    def test_gradient_flows(self):
        """Generator parameters should receive gradients."""
        labels = torch.randint(0, 10, (4,))
        result = self.gen(labels)
        loss = result["output"].sum()
        loss.backward()
        for p in self.gen.parameters():
            assert p.grad is not None


# ── FedGenModelWrapper Tests ────────────────────────────────────────


class TestFedGenModelWrapper:
    def test_wrapper_cnn_forward_shape(self):
        model = _SimpleCNN(num_classes=10)
        wrapped = FedGenModelWrapper(model, latent_dim=32, num_classes=10)
        x = torch.randn(4, 1, 28, 28)
        out = wrapped(x)
        assert out.shape == (4, 10)

    def test_wrapper_vit_forward_shape(self):
        model = _SimpleViT(num_classes=10)
        wrapped = FedGenModelWrapper(model, latent_dim=32, num_classes=10)
        x = torch.randn(4, 1, 28, 28)
        out = wrapped(x)
        assert out.shape == (4, 10)

    def test_forward_from_latent_shape(self):
        model = _SimpleCNN(num_classes=10)
        wrapped = FedGenModelWrapper(model, latent_dim=32, num_classes=10)
        z = torch.randn(4, 32)
        out = wrapped.forward_from_latent(z)
        assert out.shape == (4, 10)

    def test_forward_from_latent_uses_only_classifier(self):
        """forward_from_latent should not go through backbone."""
        model = _SimpleCNN(num_classes=10)
        wrapped = FedGenModelWrapper(model, latent_dim=32, num_classes=10)

        z = torch.randn(4, 32)
        out1 = wrapped.forward_from_latent(z)

        # Directly compute: classifier(z)
        out2 = wrapped.classifier(z)
        assert torch.allclose(out1, out2)

    def test_classifier_detection_fc(self):
        model = _SimpleCNN(num_classes=10)
        attr, in_features, _ = _find_classifier_info(model)
        assert attr == "fc"
        assert in_features == 8

    def test_classifier_detection_heads(self):
        model = _SimpleViT(num_classes=10)
        attr, in_features, _ = _find_classifier_info(model)
        assert attr == "heads"
        assert in_features == 16

    def test_wrapper_preserves_feature_dim(self):
        model = _SimpleCNN(num_classes=10)
        wrapped = FedGenModelWrapper(model, latent_dim=32, num_classes=10)
        assert wrapped.feature_dim == 8

    def test_wrapper_bottleneck_shape(self):
        model = _SimpleCNN(num_classes=10)
        wrapped = FedGenModelWrapper(model, latent_dim=64, num_classes=10)
        assert wrapped.bottleneck.in_features == 8
        assert wrapped.bottleneck.out_features == 64
        assert wrapped.classifier.in_features == 64
        assert wrapped.classifier.out_features == 10

    def test_gradient_flows_through_wrapper(self):
        model = _SimpleCNN(num_classes=10)
        wrapped = FedGenModelWrapper(model, latent_dim=32, num_classes=10)
        x = torch.randn(4, 1, 28, 28)
        out = wrapped(x)
        loss = out.sum()
        loss.backward()
        # Backbone, bottleneck, and classifier should all have gradients
        assert wrapped.bottleneck.weight.grad is not None
        assert wrapped.classifier.weight.grad is not None

    def test_gradient_flows_through_forward_from_latent(self):
        model = _SimpleCNN(num_classes=10)
        wrapped = FedGenModelWrapper(model, latent_dim=32, num_classes=10)
        z = torch.randn(4, 32, requires_grad=True)
        out = wrapped.forward_from_latent(z)
        loss = out.sum()
        loss.backward()
        assert z.grad is not None
        assert wrapped.classifier.weight.grad is not None


# ── FedGenClient Tests ──────────────────────────────────────────────


class TestFedGenClient:
    def setup_method(self):
        model = _SimpleCNN(num_classes=10)
        self.wrapped = FedGenModelWrapper(model, latent_dim=32, num_classes=10)
        self.gen = FedGenGenerator(
            noise_dim=32, num_classes=10, hidden_dim=64, latent_dim=32
        )
        self.loader = _make_loader(num_samples=64, num_classes=10, batch_size=16)

    def test_client_fit_returns_correct_structure(self):
        client = FedGenClient(
            cid="0",
            model=self.wrapped,
            generator=self.gen,
            train_loader=self.loader,
            epochs=1,
        )
        params, num_ex, metrics = client.fit([], {"round": 1})
        assert isinstance(params, list)
        assert num_ex == 64
        assert "label_counts" in metrics

    def test_client_fit_with_regularization(self):
        """Training with regularization should not crash and should update weights."""
        # Fresh model to avoid any state leakage
        model = _SimpleCNN(num_classes=10)
        wrapped = FedGenModelWrapper(model, latent_dim=32, num_classes=10)
        gen = FedGenGenerator(
            noise_dim=32, num_classes=10, hidden_dim=64, latent_dim=32
        )
        loader = _make_loader(num_samples=64, num_classes=10, batch_size=16)

        client = FedGenClient(
            cid="0",
            model=wrapped,
            generator=gen,
            train_loader=loader,
            epochs=2,
            lr=0.1,
        )
        # Capture initial state
        initial_state = {k: v.clone() for k, v in wrapped.state_dict().items()}

        # Train with regularization
        params_ret, _, _ = client.fit([], {"round": 2, "regularization": True})

        # Parameters should change after training
        current_state = wrapped.state_dict()
        changed = any(
            not torch.equal(initial_state[k], current_state[k]) for k in initial_state
        )
        assert changed, "Model parameters did not change after training"

    def test_client_fit_without_regularization(self):
        """Round 0 (no regularization) should work normally."""
        client = FedGenClient(
            cid="0",
            model=self.wrapped,
            generator=self.gen,
            train_loader=self.loader,
            epochs=1,
        )
        params, num_ex, metrics = client.fit([], {"round": 0})
        assert num_ex == 64

    def test_label_counts_computed_correctly(self):
        client = FedGenClient(
            cid="0",
            model=self.wrapped,
            generator=self.gen,
            train_loader=self.loader,
            epochs=1,
        )
        assert len(client.label_counts) == 10
        assert sum(client.label_counts) == 64

    def test_available_labels_non_empty(self):
        client = FedGenClient(
            cid="0",
            model=self.wrapped,
            generator=self.gen,
            train_loader=self.loader,
            epochs=1,
        )
        assert len(client.available_labels) > 0

    def test_evaluate_returns_correct_structure(self):
        client = FedGenClient(
            cid="0",
            model=self.wrapped,
            generator=self.gen,
            train_loader=self.loader,
            epochs=1,
        )
        loss, num_ex, metrics = client.evaluate([], {})
        assert isinstance(loss, float)
        assert num_ex == 64
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0


class TestExpLrDecay:
    def test_decay_round_0(self):
        assert exp_lr_decay(0, 10.0) == 10.0

    def test_decay_round_1(self):
        assert abs(exp_lr_decay(1, 10.0) - 10.0 * DECAY_RATE) < 1e-6

    def test_decay_round_10(self):
        expected = DEFAULT_GENERATIVE_ALPHA * (DECAY_RATE**10)
        assert abs(exp_lr_decay(10, DEFAULT_GENERATIVE_ALPHA) - expected) < 1e-6

    def test_decay_monotonically_decreasing(self):
        values = [exp_lr_decay(i, 10.0) for i in range(20)]
        for i in range(1, len(values)):
            assert values[i] < values[i - 1]


# ── Integration: Generator + Wrapper + Client ───────────────────────


class TestFedGenEndToEnd:
    def test_generator_output_feeds_into_wrapper(self):
        """Generator latent output should be directly consumable by wrapper."""
        gen = FedGenGenerator(latent_dim=32, num_classes=10)
        model = _SimpleCNN(num_classes=10)
        wrapped = FedGenModelWrapper(model, latent_dim=32, num_classes=10)

        labels = torch.randint(0, 10, (4,))
        gen_result = gen(labels)
        latent = gen_result["output"]

        # Should produce valid logits
        logits = wrapped.forward_from_latent(latent)
        assert logits.shape == (4, 10)
        assert torch.isfinite(logits).all()

    def test_two_round_pipeline(self):
        """Simulate 2 rounds of FedGen: train client, verify no crash."""
        gen = FedGenGenerator(latent_dim=32, num_classes=10, hidden_dim=64)
        model = _SimpleCNN(num_classes=10)
        wrapped = FedGenModelWrapper(model, latent_dim=32, num_classes=10)
        loader = _make_loader(num_samples=32, num_classes=10, batch_size=8)

        client = FedGenClient(
            cid="0",
            model=wrapped,
            generator=gen,
            train_loader=loader,
            epochs=2,
        )

        # Round 1: no regularization
        params, _, _ = client.fit([], {"round": 0})
        assert len(params) > 0

        # Round 2: with regularization
        params, _, _ = client.fit(params, {"round": 1})
        assert len(params) > 0
