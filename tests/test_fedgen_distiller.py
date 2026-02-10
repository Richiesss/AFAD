"""Tests for FedGenDistiller."""

import numpy as np
import torch
import torch.nn as nn

from src.server.generator.synthetic_generator import SyntheticGenerator
from src.server.strategy.fedgen_distiller import FedGenDistiller


def _make_simple_model(in_features: int = 784, num_classes: int = 10) -> nn.Module:
    """Create a small MLP for testing."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes),
    )


class TestFedGenDistiller:
    def setup_method(self):
        self.generator = SyntheticGenerator(noise_dim=32, num_classes=10, hidden_dim=64)
        self.distiller = FedGenDistiller(
            generator=self.generator,
            gen_lr=1e-3,
            batch_size=8,
            device="cpu",
        )

    def test_get_label_weights_uniform(self):
        """With uniform label distribution, weights should be equal."""
        counts = [[100] * 10, [100] * 10]
        weights, qualified = self.distiller.get_label_weights(counts)
        assert weights.shape == (10, 2)
        assert len(qualified) == 10
        np.testing.assert_allclose(weights[0], [0.5, 0.5])

    def test_get_label_weights_non_iid(self):
        """Non-IID: client 0 has all of label 0, client 1 has none."""
        counts = [
            [100, 0, 50, 50, 50, 50, 50, 50, 50, 50],
            [0, 100, 50, 50, 50, 50, 50, 50, 50, 50],
        ]
        weights, qualified = self.distiller.get_label_weights(counts)
        # Label 0: only client 0 has samples
        assert weights[0, 0] == 1.0
        assert weights[0, 1] == 0.0
        # Label 1: only client 1 has samples
        assert weights[1, 0] == 0.0
        assert weights[1, 1] == 1.0

    def test_train_generator_needs_two_models(self):
        """Generator training requires at least 2 models."""
        model = _make_simple_model()
        label_weights = np.ones((10, 1)) / 1
        loss = self.distiller.train_generator(
            {"a": model},
            label_weights,
            list(range(10)),
            num_epochs=1,
            num_teacher_iters=1,
        )
        assert loss == 0.0

    def test_train_generator_modifies_weights(self):
        """Generator weights should change during training."""
        models = {"a": _make_simple_model(), "b": _make_simple_model()}
        label_weights = np.ones((10, 2)) / 2
        qualified_labels = list(range(10))

        gen_weights_before = [p.clone() for p in self.generator.parameters()]

        self.distiller.train_generator(
            models,
            label_weights,
            qualified_labels,
            num_epochs=1,
            num_teacher_iters=5,
        )

        any_changed = False
        for p_before, p_after in zip(gen_weights_before, self.generator.parameters()):
            if not torch.equal(p_before, p_after):
                any_changed = True
                break
        assert any_changed, "Generator weights should change during training"

    def test_train_generator_returns_loss(self):
        models = {"a": _make_simple_model(), "b": _make_simple_model()}
        label_weights = np.ones((10, 2)) / 2

        loss = self.distiller.train_generator(
            models,
            label_weights,
            list(range(10)),
            num_epochs=1,
            num_teacher_iters=3,
        )
        assert isinstance(loss, float)
        assert loss > 0.0

    def test_model_params_restored_after_training(self):
        """Model parameters should have requires_grad=True after training."""
        models = {"a": _make_simple_model(), "b": _make_simple_model()}
        label_weights = np.ones((10, 2)) / 2

        self.distiller.train_generator(
            models,
            label_weights,
            list(range(10)),
            num_epochs=1,
            num_teacher_iters=2,
        )

        for model in models.values():
            for p in model.parameters():
                assert p.requires_grad, "Model params should have grad enabled"


class TestDistillModels:
    def setup_method(self):
        self.generator = SyntheticGenerator(noise_dim=32, num_classes=10, hidden_dim=64)
        self.distiller = FedGenDistiller(
            generator=self.generator,
            gen_lr=1e-3,
            batch_size=8,
            device="cpu",
            temperature=4.0,
            distill_lr=1e-3,
            distill_epochs=1,
            distill_steps=3,
            distill_alpha=0.7,
        )

    def test_distill_needs_two_models(self):
        """Distillation requires at least 2 models."""
        model = _make_simple_model()
        label_weights = np.ones((10, 1))
        result = self.distiller.distill_models(
            {"a": model}, label_weights, list(range(10))
        )
        assert result == {}

    def test_distill_modifies_weights(self):
        """Model weights should change during distillation."""
        models = {"a": _make_simple_model(), "b": _make_simple_model()}
        label_weights = np.ones((10, 2)) / 2

        # Pre-train generator so it produces non-trivial output
        self.distiller.train_generator(
            models,
            label_weights,
            list(range(10)),
            num_epochs=1,
            num_teacher_iters=3,
        )

        weights_before = {
            sig: [p.clone() for p in m.parameters()] for sig, m in models.items()
        }

        self.distiller.distill_models(
            models,
            label_weights,
            list(range(10)),
            min_quality=0.0,
        )

        any_changed = False
        for sig, model in models.items():
            for p_before, p_after in zip(weights_before[sig], model.parameters()):
                if not torch.equal(p_before, p_after):
                    any_changed = True
                    break
        assert any_changed, "At least one model should have changed weights"

    def test_distill_returns_losses(self):
        """Should return a dict of signature -> positive loss."""
        models = {"a": _make_simple_model(), "b": _make_simple_model()}
        label_weights = np.ones((10, 2)) / 2

        self.distiller.train_generator(
            models,
            label_weights,
            list(range(10)),
            num_epochs=1,
            num_teacher_iters=3,
        )

        losses = self.distiller.distill_models(
            models,
            label_weights,
            list(range(10)),
            min_quality=0.0,
        )
        assert len(losses) == 2
        for sig, loss in losses.items():
            assert isinstance(loss, float)
            assert loss > 0.0

    def test_distill_does_not_modify_generator(self):
        """Generator weights should not change during distillation."""
        models = {"a": _make_simple_model(), "b": _make_simple_model()}
        label_weights = np.ones((10, 2)) / 2

        self.distiller.train_generator(
            models,
            label_weights,
            list(range(10)),
            num_epochs=1,
            num_teacher_iters=3,
        )

        gen_weights_before = [p.clone() for p in self.generator.parameters()]

        self.distiller.distill_models(
            models,
            label_weights,
            list(range(10)),
            min_quality=0.0,
        )

        for p_before, p_after in zip(gen_weights_before, self.generator.parameters()):
            assert torch.equal(p_before, p_after), "Generator should not change"

    def test_distill_restores_requires_grad(self):
        """All model and generator params should have requires_grad=True after."""
        models = {"a": _make_simple_model(), "b": _make_simple_model()}
        label_weights = np.ones((10, 2)) / 2

        self.distiller.train_generator(
            models,
            label_weights,
            list(range(10)),
            num_epochs=1,
            num_teacher_iters=3,
        )

        self.distiller.distill_models(
            models,
            label_weights,
            list(range(10)),
            min_quality=0.0,
        )

        for model in models.values():
            for p in model.parameters():
                assert p.requires_grad, "Model params should have grad enabled"
        for p in self.generator.parameters():
            assert p.requires_grad, "Generator params should have grad enabled"
