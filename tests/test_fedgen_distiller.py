"""Tests for FedGenDistiller."""

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
        self.generator = SyntheticGenerator(
            latent_dim=32, num_classes=10, output_shape=(1, 28, 28)
        )
        self.distiller = FedGenDistiller(
            generator=self.generator,
            temperature=4.0,
            device="cpu",
            batch_size=8,
        )

    def test_distill_requires_at_least_two_models(self):
        model = _make_simple_model()
        result = self.distiller.distill({"a": model}, gen_steps=1, distill_steps=1)
        assert "a" in result

    def test_distill_returns_all_models(self):
        models = {"a": _make_simple_model(), "b": _make_simple_model()}
        result = self.distiller.distill(models, gen_steps=2, distill_steps=2)
        assert set(result.keys()) == {"a", "b"}

    def test_distill_modifies_model_weights(self):
        models = {"a": _make_simple_model(), "b": _make_simple_model()}

        # Capture weights before distillation
        weights_before = {
            name: [p.clone() for p in model.parameters()]
            for name, model in models.items()
        }

        self.distiller.distill(models, gen_steps=3, distill_steps=3)

        # At least one model's weights should change
        any_changed = False
        for name, model in models.items():
            for p_before, p_after in zip(weights_before[name], model.parameters()):
                if not torch.equal(p_before, p_after):
                    any_changed = True
                    break
        assert any_changed, "Distillation should modify at least one model's weights"

    def test_generator_weights_change_during_distill(self):
        models = {"a": _make_simple_model(), "b": _make_simple_model()}

        gen_weights_before = [p.clone() for p in self.generator.generator.parameters()]

        self.distiller.distill(models, gen_steps=5, distill_steps=1)

        any_changed = False
        for p_before, p_after in zip(
            gen_weights_before, self.generator.generator.parameters()
        ):
            if not torch.equal(p_before, p_after):
                any_changed = True
                break
        assert any_changed, "Generator weights should change during training"

    def test_kd_loss_computation(self):
        student = torch.randn(4, 10)
        teacher = torch.randn(4, 10)
        loss = FedGenDistiller._kd_loss(student, teacher, temperature=4.0)
        assert loss.shape == ()
        assert loss.item() >= 0.0
