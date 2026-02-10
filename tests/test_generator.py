"""Tests for SyntheticGenerator."""

import torch

from src.server.generator.synthetic_generator import (
    MNIST_MEAN,
    MNIST_STD,
    SyntheticGenerator,
)


class TestSyntheticGenerator:
    def setup_method(self):
        self.generator = SyntheticGenerator(
            noise_dim=32, num_classes=10, hidden_dim=256
        )

    def test_forward_returns_dict(self):
        labels = torch.randint(0, 10, (4,))
        result = self.generator(labels)
        assert "output" in result
        assert "eps" in result

    def test_forward_output_shape(self):
        labels = torch.randint(0, 10, (4,))
        result = self.generator(labels)
        assert result["output"].shape == (4, 1, 28, 28)

    def test_forward_eps_shape(self):
        labels = torch.randint(0, 10, (4,))
        result = self.generator(labels)
        assert result["eps"].shape == (4, 32)

    def test_forward_output_is_normalized(self):
        """Output should be MNIST-normalized (not raw [0,1])."""
        labels = torch.randint(0, 10, (16,))
        result = self.generator(labels)
        images = result["output"]
        expected_min = (0.0 - MNIST_MEAN) / MNIST_STD
        expected_max = (1.0 - MNIST_MEAN) / MNIST_STD
        assert torch.all(images >= expected_min - 1e-5)
        assert torch.all(images <= expected_max + 1e-5)

    def test_diversity_loss_returns_scalar(self):
        labels = torch.randint(0, 10, (8,))
        result = self.generator(labels)
        loss = self.generator.diversity_loss(result["eps"], result["output"])
        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_diversity_loss_small_batch(self):
        """Diversity loss should handle batch_size=2 (min for BatchNorm)."""
        labels = torch.randint(0, 10, (2,))
        result = self.generator(labels)
        loss = self.generator.diversity_loss(result["eps"], result["output"])
        assert loss.shape == ()

    def test_forward_eval_mode_batch_one(self):
        """In eval mode, batch_size=1 should work (BatchNorm uses running stats)."""
        self.generator.eval()
        labels = torch.randint(0, 10, (1,))
        result = self.generator(labels)
        assert result["output"].shape == (1, 1, 28, 28)

    def test_crossentropy_loss_per_sample(self):
        """crossentropy_loss returns per-sample losses (no reduction)."""
        logp = torch.randn(4, 10).log_softmax(dim=1)
        labels = torch.randint(0, 10, (4,))
        losses = SyntheticGenerator.crossentropy_loss(logp, labels)
        assert losses.shape == (4,)
        assert torch.all(losses >= 0.0)

    def test_legacy_latent_dim_parameter(self):
        """Legacy latent_dim parameter should be accepted."""
        gen = SyntheticGenerator(latent_dim=100, num_classes=10)
        assert gen.latent_dim == 100

    def test_different_batch_sizes(self):
        for batch_size in [2, 4, 16, 32]:
            labels = torch.randint(0, 10, (batch_size,))
            result = self.generator(labels)
            assert result["output"].shape == (batch_size, 1, 28, 28)
