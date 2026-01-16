from src.models.registry import ModelRegistry
from src.models.vit.vit import _create_vit

# DeiTは本来Distillation Tokenを持ちますが、AFADの異種間蒸留はLogitベースのFedGen方式であり、
# アーキテクチャ内部のDistillation Token依存ではないため、
# Phase 1では簡略化してViTと同じアーキテクチャでパラメータ規模を調整したものとして実装します。
# 必要であれば後でtimmのDeiTをラップするように変更します。

@ModelRegistry.register("deit_tiny", family="deit", complexity=0.06)
def create_deit_tiny(num_classes: int = 10, **kwargs):
    # DeiT-Tiny is basically ViT-Tiny
    return _create_vit(
        num_classes=num_classes,
        image_size=28,
        patch_size=4,
        num_layers=12,
        num_heads=3,
        hidden_dim=192,
        mlp_dim=192*4
    )

@ModelRegistry.register("deit_small", family="deit", complexity=0.26)
def create_deit_small(num_classes: int = 10, **kwargs):
    # DeiT-Small is basically ViT-Small
    return _create_vit(
        num_classes=num_classes,
        image_size=28,
        patch_size=4,
        num_layers=12,
        num_heads=6,
        hidden_dim=384,
        mlp_dim=384*4
    )
