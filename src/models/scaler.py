import torch.nn as nn


class Scaler(nn.Module):
    """HeteroFL static scaler (Diao et al., ICLR 2021).

    Compensates for reduced channel count in sub-models by dividing
    activations by the model_rate. This ensures that the expected
    magnitude of activations is preserved regardless of width.

    Applied identically during both training and evaluation.
    """

    def __init__(self, rate: float):
        super().__init__()
        self.rate = rate

    def forward(self, x):
        return x / self.rate
