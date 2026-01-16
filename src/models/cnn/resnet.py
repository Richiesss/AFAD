import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50
from src.models.registry import ModelRegistry

@ModelRegistry.register("resnet50", family="resnet", complexity=1.0)
def create_resnet50(num_classes: int = 10, **kwargs):
    model = resnet50(num_classes=num_classes)
    # MNIST用に入力チャネルを1に変更 (conv1)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model

@ModelRegistry.register("resnet18", family="resnet", complexity=0.46)
def create_resnet18(num_classes: int = 10, **kwargs):
    model = resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model

@ModelRegistry.register("resnet34", family="resnet", complexity=0.85)
def create_resnet34(num_classes: int = 10, **kwargs):
    model = resnet34(num_classes=num_classes)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model
