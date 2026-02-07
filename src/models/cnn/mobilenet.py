import torch.nn as nn
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small

from src.models.registry import ModelRegistry


@ModelRegistry.register("mobilenetv3_large", family="mobilenet", complexity=0.21)
def create_mobilenet_large(num_classes: int = 10, **kwargs):
    model = mobilenet_v3_large(num_classes=num_classes)
    # MNIST用にConv1を変更
    # MobileNetV3のfeatures[0][0]はConv2d
    model.features[0][0] = nn.Conv2d(
        1, 16, kernel_size=3, stride=2, padding=1, bias=False
    )
    return model


@ModelRegistry.register(
    "mobilenetv3_small", family="mobilenet", complexity=0.10
)  # arbitrary complexity
def create_mobilenet_small(num_classes: int = 10, **kwargs):
    model = mobilenet_v3_small(num_classes=num_classes)
    model.features[0][0] = nn.Conv2d(
        1, 16, kernel_size=3, stride=2, padding=1, bias=False
    )
    return model
