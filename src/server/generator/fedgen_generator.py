"""Faithful FedGen generator (Zhu et al., ICML 2021).

Outputs latent representations (NOT images) that are fed into client
models at an intermediate layer. This matches the original paper's
architecture exactly.

Original config for MNIST:
  noise_dim=32, num_classes=10, hidden_dim=256, latent_dim=32
  Architecture: noise(32) + one_hot(10) → FC(256) + BN + ReLU → Linear(32)

Rate-conditioned extension:
  When rate_conditioned=True, the generator accepts a `rate` argument and
  produces rate-specific latents via a shared trunk + per-rate output heads.
  Architecture: noise(32) + one_hot(C) + rate_emb(8) + log(rate)(1)
                → FC(256) + BN + ReLU → head[rate_idx](256, 32)
"""

import math

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
        rate_conditioned: If True, conditions output on HeteroFL width rate.
            Adds a rate embedding and separate output head per rate value.
            Backward compatible: rate=None falls back to rate=1.0 head.
        rate_embed_dim: Embedding dimension for rate index (rate_conditioned only).
    """

    RATE_VALUES: tuple[float, ...] = (1.0, 0.5, 0.25)

    def __init__(
        self,
        noise_dim: int = 32,
        num_classes: int = 10,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        rate_conditioned: bool = False,
        rate_embed_dim: int = 8,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.rate_conditioned = rate_conditioned

        if rate_conditioned:
            self.rate_embedding = nn.Embedding(len(self.RATE_VALUES), rate_embed_dim)
            # input: noise + one_hot(y) + rate_emb + log(rate)
            input_dim = noise_dim + num_classes + rate_embed_dim + 1
        else:
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

        if rate_conditioned:
            # Separate output head per rate for rate-specific latent geometry
            self.heads = nn.ModuleList(
                [nn.Linear(hidden_dim, latent_dim) for _ in self.RATE_VALUES]
            )
        else:
            # Output: latent representation (NOT image)
            self.representation_layer = nn.Linear(hidden_dim, latent_dim)

    def _rate_to_idx(self, rate: float) -> int:
        """Map a rate float to its index in RATE_VALUES."""
        for i, r in enumerate(self.RATE_VALUES):
            if abs(r - rate) < 1e-6:
                return i
        return 0  # fallback to rate=1.0

    def forward(
        self,
        labels: torch.Tensor,
        rate: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Generate latent representations conditional on labels (and optionally rate).

        Args:
            labels: [Batch] integer class labels.
            rate: HeteroFL width rate (1.0 / 0.5 / 0.25). Only used when
                rate_conditioned=True. When None, defaults to rate=1.0.

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

        if self.rate_conditioned:
            effective_rate = rate if rate is not None else 1.0
            rate_idx = self._rate_to_idx(effective_rate)
            rate_idx_t = torch.full(
                (batch_size,), rate_idx, dtype=torch.long, device=device
            )
            rate_emb = self.rate_embedding(rate_idx_t)  # [B, rate_embed_dim]
            log_rate = torch.full(
                (batch_size, 1), math.log(effective_rate), device=device
            )
            z = torch.cat([eps, y_input, rate_emb, log_rate], dim=1)
        else:
            z = torch.cat([eps, y_input], dim=1)

        for layer in self.fc_layers:
            z = layer(z)

        if self.rate_conditioned:
            effective_rate = rate if rate is not None else 1.0
            rate_idx = self._rate_to_idx(effective_rate)
            z = self.heads[rate_idx](z)
        else:
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
