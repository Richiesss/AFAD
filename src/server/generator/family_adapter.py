"""Family-specific latent space adapters for AFAD.

Each family (e.g., CNN, ViT) learns a small residual MLP that maps the
generator's unified latent output to the family's own latent space.

Problem: FedGenGenerator is trained on full-rate family models, but sub-rate
clients develop different latent representations through their narrower
bottleneck layers. Over rounds, the generator's latent space drifts away
from what sub-rate clients and cross-family models expect.

Solution: Per-family FamilyAdapter (32→64→32, residual) trained server-side
after each generator update. Clients receive their family's adapter and apply
it before forward_from_latent(), bridging the generator↔classifier gap.

The residual connection initialises the adapter as near-identity, preventing
catastrophic changes in early rounds.
"""

import pickle
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


class FamilyAdapter(nn.Module):
    """Small residual MLP adapter for one model family.

    Maps generator latent → family-aligned latent:
        z_adapted = net(z) + z   (residual)

    The residual ensures the adapter starts as near-identity and only
    learns meaningful deviations as training progresses.

    Args:
        latent_dim: Latent space dimension (must match generator output).
        hidden_multiplier: Hidden layer width = latent_dim * hidden_multiplier.
    """

    def __init__(self, latent_dim: int = 32, hidden_multiplier: int = 2) -> None:
        super().__init__()
        hidden_dim = latent_dim * hidden_multiplier
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply adapter with residual connection."""
        return self.net(z) + z


class FamilyAdapterBank:
    """Collection of per-family adapters managed on the server.

    Each family gets one FamilyAdapter. The bank handles serialization
    for client distribution and keeps adapters on the training device.

    Args:
        families: List of family names (e.g., ["cnn", "vit"]).
        latent_dim: Latent dimension (must match generator output).
        device: Training device for adapter parameters.
    """

    def __init__(
        self,
        families: list[str],
        latent_dim: int = 32,
        device: str = "cpu",
    ) -> None:
        self.latent_dim = latent_dim
        self.device = device
        self.adapters: dict[str, FamilyAdapter] = {
            family: FamilyAdapter(latent_dim=latent_dim).to(device)
            for family in families
        }

    def get_families(self) -> list[str]:
        return list(self.adapters.keys())

    def parameters(self, family: str):
        """Yield parameters for a single family adapter."""
        return self.adapters[family].parameters()

    def train_mode(self) -> None:
        for adapter in self.adapters.values():
            adapter.train()

    def eval_mode(self) -> None:
        for adapter in self.adapters.values():
            adapter.eval()

    def serialize_family(self, family: str) -> bytes:
        """Serialize one family adapter's params as bytes for client distribution."""
        if family not in self.adapters:
            return b""
        params = [v.cpu().numpy() for v in self.adapters[family].state_dict().values()]
        return pickle.dumps(params)

    @staticmethod
    def deserialize(data: bytes, latent_dim: int = 32) -> "FamilyAdapter":
        """Reconstruct a FamilyAdapter from serialized bytes."""
        params: list[np.ndarray] = pickle.loads(data)  # noqa: S301
        adapter = FamilyAdapter(latent_dim=latent_dim)
        keys = list(adapter.state_dict().keys())
        state_dict = OrderedDict(
            (k, torch.tensor(p)) for k, p in zip(keys, params)
        )
        adapter.load_state_dict(state_dict)
        return adapter
