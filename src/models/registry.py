from typing import Callable, Dict, List, Optional
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class ModelInfo:
    name: str
    family: str
    complexity_level: float = 1.0  # Default to max complexity if unknown

class ModelRegistry:
    """
    モデルの登録と生成を管理するクラス
    """
    _models: Dict[str, Callable[..., nn.Module]] = {}
    _family_mapping: Dict[str, str] = {}
    _info: Dict[str, ModelInfo] = {}

    @classmethod
    def register(cls, name: str, family: str, complexity: float = 1.0):
        """
        モデル登録用のデコレータ
        """
        def decorator(model_factory: Callable[..., nn.Module]):
            cls._models[name] = model_factory
            cls._family_mapping[name] = family
            cls._info[name] = ModelInfo(name, family, complexity)
            return model_factory
        return decorator

    @classmethod
    def create_model(cls, name: str, num_classes: int = 10, **kwargs) -> nn.Module:
        """
        指定された名前のモデルを生成する
        """
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not found in registry.")
            
        factory = cls._models[name]
        return factory(num_classes=num_classes, **kwargs)

    @classmethod
    def get_model_info(cls, name: str) -> Optional[ModelInfo]:
        return cls._info.get(name)

    @classmethod
    def get_available_models(cls, family: Optional[str] = None) -> List[str]:
        if family:
            return [name for name, f in cls._family_mapping.items() if f == family]
        return list(cls._models.keys())
