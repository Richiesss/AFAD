"""Client-side projection head for AFAD latent space alignment.

Maps the effective latent representation of sub-rate clients (which have
fewer active dimensions due to HeteroFL width scaling) to the full 32-dim
space of the generator, enabling dimension-matched KD.

HeteroFL sets sub-model weights by slicing:
  rate=1.0: bottleneck rows 0-31 active  → effective_dim=32
  rate=0.5: bottleneck rows 0-15 active  → effective_dim=16
  rate=0.25: bottleneck rows 0-7 active  → effective_dim=8

The projection head P_r: z_{eff} → z_{32} learns to align the
client's compressed latent geometry with the generator's target space.
"""

import torch.nn as nn


class ProjectionHead(nn.Module):
    """Linear projection from effective client latent dim to generator dim.

    For rate=1.0 clients (effective_dim == latent_dim), this is a square
    linear map that learns geometric alignment (no dimension change).
    For sub-rate clients, this up-projects the effective low-dim latent
    to match the generator's output dimension.

    Args:
        effective_dim: Number of active latent dimensions for this rate.
            Computed as int(latent_dim * model_rate).
        latent_dim: Generator output dimension (full rate, typically 32).
    """

    def __init__(self, effective_dim: int, latent_dim: int = 32) -> None:
        super().__init__()
        self.effective_dim = effective_dim
        self.latent_dim = latent_dim
        self.proj = nn.Linear(effective_dim, latent_dim, bias=False)

    def forward(self, z_eff):
        """Project effective latent to full generator latent space.

        Args:
            z_eff: Effective latent [B, effective_dim] (already sliced).

        Returns:
            Projected latent [B, latent_dim].
        """
        return self.proj(z_eff)
