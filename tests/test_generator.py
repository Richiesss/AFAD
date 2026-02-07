import unittest

import torch

from src.server.generator.synthetic_generator import SyntheticGenerator


class TestSyntheticGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = SyntheticGenerator(
            latent_dim=100, num_classes=10, output_shape=(1, 28, 28)
        )

    def test_forward_shape(self):
        batch_size = 4
        z = torch.randn(batch_size, 100)
        labels = torch.randint(0, 10, (batch_size,))

        out = self.generator(z, labels)
        self.assertEqual(out.shape, (batch_size, 1, 28, 28))

    def test_generate_batch(self):
        batch_size = 8
        out, labels = self.generator.generate_batch(batch_size, device="cpu")
        self.assertEqual(out.shape, (batch_size, 1, 28, 28))
        self.assertEqual(labels.shape, (batch_size,))
        self.assertTrue(torch.all(out >= 0.0))
        self.assertTrue(torch.all(out <= 1.0))


if __name__ == "__main__":
    unittest.main()
