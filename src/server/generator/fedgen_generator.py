"""Faithful FedGen generator (Zhu et al., ICML 2021).

Outputs latent representations (NOT images) that are fed into client
models at an intermediate layer. This matches the original paper's
architecture exactly.

Original config for MNIST:
  noise_dim=32, num_classes=10, hidden_dim=256, latent_dim=32
  Architecture: noise(32) + one_hot(10) → FC(256) + BN + ReLU → Linear(32)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FedGenGenerator(nn.Module):
    """FedGen conditional generator that produces latent representations.

    Unlike SyntheticGenerator (which produces images), this generator
    outputs latent vectors that are fed directly into the model's
    classifier layer, bypassing the feature extractor.

    Args:
        noise_dim: Dimension of input noise vector.
        num_classes: Number of label classes (for one-hot conditioning).
        hidden_dim: Hidden layer dimension.
        latent_dim: Output latent representation dimension.
    """

    def __init__(
        self,
        noise_dim: int = 32,
        num_classes: int = 10,
        hidden_dim: int = 256,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        input_dim = noise_dim + num_classes

        self.fc_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                ),
            ]
        )

        # Output: latent representation (NOT image)
        self.representation_layer = nn.Linear(hidden_dim, latent_dim)

    def forward(self, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        """Generate latent representations conditional on labels.

        Args:
            labels: [Batch] integer class labels.

        Returns:
            Dict with:
                'output': Latent representations [Batch, latent_dim]
                'eps': Noise vectors used [Batch, noise_dim]
        """
        batch_size = labels.shape[0]
        device = labels.device

        eps = torch.rand(batch_size, self.noise_dim, device=device)

        y_input = torch.zeros(batch_size, self.num_classes, device=device)
        y_input.scatter_(1, labels.view(-1, 1), 1)

        z = torch.cat([eps, y_input], dim=1)

        for layer in self.fc_layers:
            z = layer(z)
        z = self.representation_layer(z)

        return {"output": z, "eps": eps}

    def diversity_loss(
        self, eps: torch.Tensor, gen_output: torch.Tensor
    ) -> torch.Tensor:
        """Diversity loss to prevent mode collapse (same as original).

        Formula: exp(-mean(pairwise_dist(outputs) * pairwise_dist(noise)))
        """
        d_gen = F.pdist(gen_output, p=2)
        d_eps = F.pdist(eps, p=2)

        if d_gen.numel() == 0:
            return torch.tensor(0.0, device=gen_output.device)

        return torch.exp(-torch.mean(d_gen * d_eps))

    @staticmethod
    def crossentropy_loss(logp: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Per-sample cross-entropy loss (NLLLoss without reduction)."""
        return F.nll_loss(logp, labels, reduction="none")
