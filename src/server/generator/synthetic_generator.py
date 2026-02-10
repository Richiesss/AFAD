import torch
import torch.nn as nn
import torch.nn.functional as F

# MNIST normalization constants
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class SyntheticGenerator(nn.Module):
    """
    FedGen-style conditional generator for Data-Free Knowledge Distillation.

    Adapted for heterogeneous architectures: outputs MNIST images (1, 28, 28)
    instead of latent representations, so any model architecture can consume
    the generated data directly.

    Reference: Zhu et al., "Data-Free Knowledge Distillation for Heterogeneous
    Federated Learning" (ICML 2021)
    """

    def __init__(
        self,
        noise_dim: int = 32,
        num_classes: int = 10,
        hidden_dim: int = 256,
        output_shape: tuple = (1, 28, 28),
        # Legacy parameter kept for backward compatibility
        latent_dim: int | None = None,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.output_shape = output_shape
        # Legacy alias
        self.latent_dim = latent_dim if latent_dim is not None else noise_dim

        # Input: noise + one-hot label
        input_dim = noise_dim + num_classes

        # FC layers
        self.fc_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                ),
            ]
        )

        # Second hidden layer
        self.fc_layers.append(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
            )
        )

        # Output layer: project to image space
        output_dim = 1
        for d in output_shape:
            output_dim *= d
        self.representation_layer = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Generate synthetic images conditional on labels.

        Args:
            labels: [Batch] integer class labels

        Returns:
            Dict with:
                'output': Generated images [Batch, C, H, W], MNIST-normalized
                'eps': Noise vectors used [Batch, noise_dim]
        """
        batch_size = labels.shape[0]
        device = labels.device

        # Random noise (uniform, matching FedGen paper)
        eps = torch.rand(batch_size, self.noise_dim, device=device)

        # One-hot label encoding
        y_input = torch.zeros(batch_size, self.num_classes, device=device)
        y_input.scatter_(1, labels.view(-1, 1), 1)

        z = torch.cat([eps, y_input], dim=1)

        # Forward through FC layers
        for layer in self.fc_layers:
            z = layer(z)
        z = self.representation_layer(z)

        # Sigmoid to [0, 1] range, then MNIST normalization
        z = torch.sigmoid(z)
        images = z.view(-1, *self.output_shape)
        images = (images - MNIST_MEAN) / MNIST_STD

        return {"output": images, "eps": eps}

    def diversity_loss(
        self, eps: torch.Tensor, gen_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Diversity loss to prevent mode collapse.

        Encourages different noise inputs to produce different outputs.
        Formula: exp(-mean(pairwise_dist(outputs) * pairwise_dist(noise)))

        Args:
            eps: Noise vectors [Batch, noise_dim]
            gen_output: Generated images [Batch, C, H, W]

        Returns:
            Scalar diversity loss
        """
        gen_flat = gen_output.reshape(gen_output.size(0), -1)
        d_gen = F.pdist(gen_flat, p=2)
        d_eps = F.pdist(eps, p=2)

        # Avoid numerical issues
        if d_gen.numel() == 0:
            return torch.tensor(0.0, device=gen_output.device)

        return torch.exp(-torch.mean(d_gen * d_eps))

    @staticmethod
    def crossentropy_loss(logp: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Per-sample cross-entropy loss (NLLLoss without reduction).

        Args:
            logp: Log-probabilities [Batch, num_classes]
            labels: Integer labels [Batch]

        Returns:
            Per-sample losses [Batch]
        """
        return F.nll_loss(logp, labels, reduction="none")
