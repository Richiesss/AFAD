import torch.nn as nn
from typing import Tuple, List, Dict, Type

class FamilyRouter:
    """
    モデルアーキテクチャのファミリー分類と管理
    """
    
    # シグネチャ定義 (簡易版)
    # 本来はグラフ解析などが望ましいが、今回はモジュール名の存在確認で行う
    SIGNATURES = {
        "resnet": ["Conv2d", "BatchNorm2d", "ReLU", "Bottleneck", "BasicBlock"],
        "mobilenet": ["Conv2d", "BatchNorm2d", "ReLU", "InvertedResidual"],
        "vit": ["PatchEmbed", "MultiheadAttention", "LayerNorm", "MLP"],
        "deit": ["PatchEmbed", "Attention", "LayerNorm", "DistilledVisionTransformer"] # Pseudo
    }
    
    FAMILY_GROUPS = {
        "cnn": ["resnet", "mobilenet", "vgg", "efficientnet"],
        "transformer": ["vit", "deit", "swin"]
    }

    def __init__(self):
        pass

    def detect_family(self, model: nn.Module) -> Tuple[str, str]:
        """
        モデルのファミリーを検出する
        
        Args:
            model: PyTorchモデル
            
        Returns:
            Tuple[str, str]: (specific_family, broad_group)
        """
        # モデル内の全モジュール型名を取得
        module_types = set()
        for m in model.modules():
            module_types.add(type(m).__name__)
            # Also check for specific layer names if needed
        
        # マッチングスコア計算
        best_family = "unknown"
        max_score = 0
        
        for family, signatures in self.SIGNATURES.items():
            score = sum(1 for sig in signatures if self._check_signature(model, sig, module_types))
            if score > max_score:
                max_score = score
                best_family = family
        
        # グループ判定
        group = "unknown"
        for g, families in self.FAMILY_GROUPS.items():
            if best_family in families:
                group = g
                break
                
        return best_family, group

    def _check_signature(self, model: nn.Module, signature: str, module_types: set) -> bool:
        """
        シグネチャ（クラス名や構造）が含まれているか確認
        """
        # 単純なクラス名一致
        if signature in module_types:
            return True
            
        # 再帰的な構造チェック（今回は簡易実装として省略し、文字列マッチのみ）
        # 例: 文字列が部分一致するか (e.g. "resnet" in repr(model))
        return False

    def get_complexity_level(self, model: nn.Module) -> float:
        """
        モデルの複雑度レベルを計算 (パラメータ数ベース)
        """
        params = sum(p.numel() for p in model.parameters())
        # 基準となる最大モデル（ResNet50 / ViT-Base）のパラメータ数を仮定
        # ResNet50: ~25.6M
        # ViT-Base: ~86M
        
        # 本来はRegistryから最大値を取得すべきだが、簡易的にハードコードまたは動的取得
        # ここでは単純に正規化せず、パラメータ数をそのまま返すか、対数スケールにするか
        # 仕様書では "level = params(model) / params(family_max_model)" とある
        
        # TODO: FamilyごとのMaxパラメータ数を定義して計算
        # いったんパラメータ数をそのままスケーリングして返す (仮のロジック)
        max_params = 26_000_000 # ResNet50 approx
        
        level = min(1.0, params / max_params)
        return float(level)
