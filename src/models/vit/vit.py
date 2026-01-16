import torch
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer
from functools import partial
from src.models.registry import ModelRegistry

def _create_vit(
    num_classes: int,
    image_size: int = 28,
    patch_size: int = 4,
    num_layers: int = 12,
    num_heads: int = 12,
    hidden_dim: int = 768,
    mlp_dim: int = 3072,
    dropout: float = 0.0,
    attention_dropout: float = 0.0,
):
    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        dropout=dropout,
        attention_dropout=attention_dropout,
        num_classes=num_classes,
    )
    # MNIST is 1 channel. Conv2d projection needs 1 input channel.
    # torchvision ViT uses Conv2d for patch embedding.
    # model.conv_proj is the patch embedding layer.
    original_proj = model.conv_proj
    model.conv_proj = nn.Conv2d(
        in_channels=1,
        out_channels=original_proj.out_channels,
        kernel_size=original_proj.kernel_size,
        stride=original_proj.stride,
        bias=original_proj.bias is not None
    )
    return model

@ModelRegistry.register("vit_tiny", family="vit", complexity=0.06)
def create_vit_tiny(num_classes: int = 10, **kwargs):
    # ViT-Tiny config approx: layers=12, hidden=192, heads=3
    return _create_vit(
        num_classes=num_classes,
        image_size=28,
        patch_size=4,
        num_layers=12,
        num_heads=3,
        hidden_dim=192,
        mlp_dim=192*4
    )

@ModelRegistry.register("vit_small", family="vit", complexity=0.26)
def create_vit_small(num_classes: int = 10, **kwargs):
    # ViT-Small config approx: layers=12, hidden=384, heads=6
    return _create_vit(
        num_classes=num_classes,
        image_size=28,
        patch_size=4,
        num_layers=12,
        num_heads=6,
        hidden_dim=384,
        mlp_dim=384*4
    )
