"""Width-scalable ViT-Small for HeteroFL (Diao et al., ICLR 2021).

Supports model_rate-based hidden dimension scaling:
  rate=1.0  -> hidden_dim=384, heads=6, mlp_dim=1536  (~21.3M params)
  rate=0.5  -> hidden_dim=192, heads=6, mlp_dim=768   (~5.4M params)
  rate=0.25 -> hidden_dim=96,  heads=6, mlp_dim=384   (~1.4M params)

Scaler is applied after conv_proj (patch embedding). LayerNorm inside
transformer blocks normalizes activations, so per-block scaling is
not needed.
"""

import torch
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer

from src.models.registry import ModelRegistry
from src.models.scaler import Scaler

BASE_HIDDEN_DIM = 384
NUM_HEADS = 6
NUM_LAYERS = 12


class HeteroFLViT(nn.Module):
    """Width-scalable ViT with HeteroFL Scaler.

    Args:
        num_classes: Number of output classes.
        model_rate: Width scaling factor (1.0 = full, 0.5 = half, etc.).
        image_size: Input image size (28 for MNIST/MedMNIST).
        patch_size: Patch size for tokenization.
    """

    def __init__(
        self,
        num_classes: int = 10,
        model_rate: float = 1.0,
        image_size: int = 28,
        patch_size: int = 4,
    ):
        super().__init__()
        self.model_rate = model_rate

        hidden_dim = int(BASE_HIDDEN_DIM * model_rate)
        mlp_dim = hidden_dim * 4

        self.vit = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            num_classes=num_classes,
        )
        # Patch conv_proj for 1-channel input (MNIST/MedMNIST)
        self.vit.conv_proj = nn.Conv2d(
            in_channels=1,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.scaler = Scaler(model_rate)

    def forward(self, x):
        # Patch embedding + scaler
        x = self.vit.conv_proj(x)  # [B, hidden_dim, H/P, W/P]
        x = self.scaler(x)
        # Reshape to sequence: [B, hidden_dim, H/P, W/P] -> [B, seq_len, hidden_dim]
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)  # [B, seq_len, hidden_dim]

        # Prepend class token and add pos embedding (from vit internals)
        cls_token = self.vit.class_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.vit.encoder.pos_embedding

        # Dropout + encoder blocks + ln
        x = self.vit.encoder.dropout(x)
        x = self.vit.encoder.layers(x)
        x = self.vit.encoder.ln(x)

        # Classification head on [CLS] token
        x = x[:, 0]
        x = self.vit.heads(x)
        return x


@ModelRegistry.register("heterofl_vit_small", family="vit", complexity=1.0)
def create_heterofl_vit_small(
    num_classes: int = 10, model_rate: float = 1.0, **kwargs
) -> nn.Module:
    return HeteroFLViT(
        num_classes=num_classes,
        model_rate=model_rate,
        image_size=kwargs.get("image_size", 28),
        patch_size=kwargs.get("patch_size", 4),
    )
