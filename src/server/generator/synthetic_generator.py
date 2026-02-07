import copy

import torch
import torch.nn as nn

# MNIST normalization constants
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class SyntheticGenerator(nn.Module):
    """
    Data-Free KD synthetic image generator for FedGen distillation.

    Generates MNIST-like images (1, 28, 28) that can be fed to any CNN/ViT model.
    Output is normalized to match MNIST distribution (mean=0.1307, std=0.3081).
    """

    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 10,
        output_shape: tuple = (1, 28, 28),
        hidden_dims: list[int] | None = None,
        ema_decay: float = 0.9,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 512]
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.output_shape = output_shape
        self.ema_decay = ema_decay

        # Label Embedding
        self.label_embed = nn.Embedding(num_classes, latent_dim)

        # Generator Network
        layers: list[nn.Module] = []
        input_dim = latent_dim * 2  # Noise + Label

        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            input_dim = h_dim

        # Final projection to image size
        output_dim = 1
        for d in output_shape:
            output_dim *= d

        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.Sigmoid())  # Output in [0, 1]

        self.generator = nn.Sequential(*layers)

        # EMA Model
        self.ema_generator = copy.deepcopy(self.generator)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Generate synthetic data (raw [0,1] range, no MNIST normalization).

        Args:
            z: [Batch, Latent] noise tensor
            labels: [Batch] class labels

        Returns:
            Generated images [Batch, C, H, W] in [0, 1] range
        """
        c = self.label_embed(labels)
        x = torch.cat([z, c], dim=1)
        out = self.generator(x)
        return out.view(-1, *self.output_shape)

    @torch.no_grad()
    def update_ema(self) -> None:
        for p, ema_p in zip(
            self.generator.parameters(), self.ema_generator.parameters()
        ):
            ema_p.data = self.ema_decay * ema_p.data + (1 - self.ema_decay) * p.data

    def generate_batch(
        self, batch_size: int, device: str = "cpu", use_ema: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of synthetic images with MNIST normalization applied.

        Returns:
            Tuple of (images, labels) where images are normalized to match
            MNIST distribution (mean=0.1307, std=0.3081).
        """
        z = torch.randn(batch_size, self.latent_dim).to(device)
        labels = torch.randint(0, self.num_classes, (batch_size,)).to(device)

        c = self.label_embed(labels)
        x = torch.cat([z, c], dim=1)

        if use_ema:
            out = self.ema_generator(x)
        else:
            out = self.generator(x)

        images = out.view(-1, *self.output_shape)

        # Apply MNIST normalization so generated images match trained model expectations
        images = (images - MNIST_MEAN) / MNIST_STD

        return images, labels
