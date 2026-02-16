import torch.nn as nn


class Scaler(nn.Module):
    """HeteroFL static scaler (Diao et al., ICLR 2021).

    Compensates for reduced channel count in sub-models by dividing
    activations by the model_rate during training. Identity during eval.

    This matches the original paper implementation where scaling is
    paired with BatchNorm recomputation after aggregation.
    """

    def __init__(self, rate: float):
        super().__init__()
        self.rate = rate

    def forward(self, x):
        return x / self.rate if self.training else x
