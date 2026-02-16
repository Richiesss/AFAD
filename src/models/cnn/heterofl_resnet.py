"""Width-scalable ResNet18 for HeteroFL (Diao et al., ICLR 2021).

Supports model_rate-based channel width scaling:
  rate=1.0  -> channels [64, 128, 256, 512]  (~11.2M params)
  rate=0.5  -> channels [32,  64, 128, 256]  (~2.8M params)
  rate=0.25 -> channels [16,  32,  64, 128]  (~0.7M params)

A Scaler module is inserted after each residual block to compensate
for the reduced channel count: output = output / model_rate.
"""

import torch.nn as nn

from src.models.registry import ModelRegistry
from src.models.scaler import Scaler

BASE_CHANNELS = [64, 128, 256, 512]


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock (no bottleneck)."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class HeteroFLResNet(nn.Module):
    """Width-scalable ResNet18 with HeteroFL Scaler.

    Args:
        num_classes: Number of output classes.
        model_rate: Width scaling factor (1.0 = full, 0.5 = half, etc.).
        in_channels: Number of input image channels (1 for MNIST/MedMNIST).
    """

    def __init__(
        self,
        num_classes: int = 10,
        model_rate: float = 1.0,
        in_channels: int = 1,
    ):
        super().__init__()
        self.model_rate = model_rate

        channels = [int(c * model_rate) for c in BASE_CHANNELS]
        self.scaler = Scaler(model_rate)

        # Initial convolution
        self.conv1 = nn.Conv2d(
            in_channels, channels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)

        # Residual layers: [2, 2, 2, 2] blocks (ResNet18 config)
        self.layer1 = self._make_layer(channels[0], channels[0], num_blocks=2, stride=1)
        self.layer2 = self._make_layer(channels[0], channels[1], num_blocks=2, stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], num_blocks=2, stride=2)
        self.layer4 = self._make_layer(channels[2], channels[3], num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # FC layer is NOT width-scaled: always maps to num_classes
        self.fc = nn.Linear(channels[3], num_classes)

    @staticmethod
    def _make_layer(
        in_channels: int, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        current_in = in_channels
        for s in strides:
            layers.append(BasicBlock(current_in, out_channels, stride=s))
            current_in = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.scaler(self.layer1(out))
        out = self.scaler(self.layer2(out))
        out = self.scaler(self.layer3(out))
        out = self.scaler(self.layer4(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


@ModelRegistry.register("heterofl_resnet18", family="cnn", complexity=1.0)
def create_heterofl_resnet18(
    num_classes: int = 10, model_rate: float = 1.0, **kwargs
) -> nn.Module:
    return HeteroFLResNet(
        num_classes=num_classes,
        model_rate=model_rate,
        in_channels=kwargs.get("in_channels", 1),
    )
