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
            latent_dim=100, num_classes=10, output_shape=(1, 28, 28)
        )

    def test_forward_shape(self):
        batch_size = 4
        z = torch.randn(batch_size, 100)
        labels = torch.randint(0, 10, (batch_size,))
        out = self.generator(z, labels)
        assert out.shape == (batch_size, 1, 28, 28)

    def test_forward_raw_range(self):
        """forward() outputs in [0, 1] (no MNIST normalization)."""
        z = torch.randn(8, 100)
        labels = torch.randint(0, 10, (8,))
        out = self.generator(z, labels)
        assert torch.all(out >= 0.0)
        assert torch.all(out <= 1.0)

    def test_generate_batch_shape(self):
        out, labels = self.generator.generate_batch(8, device="cpu")
        assert out.shape == (8, 1, 28, 28)
        assert labels.shape == (8,)

    def test_generate_batch_is_normalized(self):
        """generate_batch() applies MNIST normalization."""
        out, _ = self.generator.generate_batch(16, device="cpu")
        # After normalization: min = (0 - 0.1307) / 0.3081, max = (1 - 0.1307) / 0.3081
        expected_min = (0.0 - MNIST_MEAN) / MNIST_STD
        expected_max = (1.0 - MNIST_MEAN) / MNIST_STD
        assert torch.all(out >= expected_min - 1e-5)
        assert torch.all(out <= expected_max + 1e-5)

    def test_ema_update(self):
        """EMA update should change ema_generator weights."""
        ema_before = [p.clone() for p in self.generator.ema_generator.parameters()]
        # Modify generator weights
        for p in self.generator.generator.parameters():
            p.data += 1.0
        self.generator.update_ema()
        for p_before, p_after in zip(
            ema_before, self.generator.ema_generator.parameters()
        ):
            assert not torch.equal(p_before, p_after)

    def test_generate_batch_ema(self):
        """generate_batch with use_ema=True should work."""
        out, labels = self.generator.generate_batch(4, device="cpu", use_ema=True)
        assert out.shape == (4, 1, 28, 28)
