"""AFAD server-side generator trainer.

Trains a FedGenGenerator using family models' forward_from_latent(),
enabling cross-family knowledge sharing through a shared latent space.

Unlike FedGenDistiller (image-based, server-side KD), this:
- Uses FedGenGenerator (latent-space output, not images)
- Uses forward_from_latent (bypasses backbone, classifier only)
- No server-side distillation (KD happens client-side via AFADClient)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.server.generator.fedgen_generator import FedGenGenerator
from src.utils.logger import setup_logger

logger = setup_logger("AFADGeneratorTrainer")

# Minimum samples per label to include in generator training
MIN_SAMPLES_PER_LABEL = 1


class EMATracker:
    """Exponential Moving Average tracker for model parameters.

    Maintains a shadow copy of parameters updated as:
        shadow = decay * shadow + (1 - decay) * current

    The EMA version is smoother and produces more stable generator outputs,
    which is beneficial when distributing to clients across rounds.

    Args:
        model: Model whose parameters to track.
        decay: EMA decay rate. Higher values = slower update, more stable.
            Typical range: 0.99 - 0.9999. Default 0.999.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self, model: nn.Module) -> None:
        """Update shadow parameters with current model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def get_state_dict(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Build a state dict using EMA weights for tracked params.

        Starts from model.state_dict() to preserve parameter ordering,
        then replaces tracked parameter values with their EMA shadows.
        Non-parameter buffers (e.g. BatchNorm running stats) are left
        unchanged since they are not tracked by EMA.
        """
        sd = model.state_dict()  # OrderedDict — preserves canonical order
        for name in sd:
            if name in self.shadow:
                sd[name] = self.shadow[name].clone()
        return sd


class AFADGeneratorTrainer:
    """Server-side FedGen generator training for AFAD.

    Each round, the server reconstructs full-rate FedGenModelWrapper
    instances from family_global_models and trains the shared generator
    so its latent outputs produce correct predictions via each family
    model's classifier (forward_from_latent).

    An optional EMA tracker smooths the generator parameters over rounds,
    producing more stable latent representations for client-side KD.

    Args:
        generator: FedGenGenerator instance.
        gen_lr: Generator learning rate.
        batch_size: Batch size for training.
        ensemble_alpha: Teacher loss weight.
        ensemble_eta: Diversity loss weight.
        device: Training device.
        ema_decay: EMA decay rate for generator parameters (0.0 = disabled).
    """

    def __init__(
        self,
        generator: FedGenGenerator,
        gen_lr: float = 3e-4,
        batch_size: int = 128,
        ensemble_alpha: float = 1.0,
        ensemble_eta: float = 1.0,
        device: str = "cpu",
        ema_decay: float = 0.999,
    ):
        self.generator = generator
        self.batch_size = batch_size
        self.ensemble_alpha = ensemble_alpha
        self.ensemble_eta = ensemble_eta
        self.device = device
        self.ema_decay = ema_decay
        self.generator.to(device)
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=gen_lr)
        self.ema_tracker = (
            EMATracker(generator, decay=ema_decay) if ema_decay > 0 else None
        )

    def get_inference_state_dict(self) -> dict[str, torch.Tensor]:
        """Return generator state dict for client distribution.

        When EMA is enabled, returns EMA-smoothed weights (more stable).
        Otherwise returns the current training weights.
        """
        if self.ema_tracker is not None:
            return self.ema_tracker.get_state_dict(self.generator)
        return self.generator.state_dict()

    @staticmethod
    def get_label_weights(
        family_label_counts: list[list[int]],
        num_classes: int = 10,
    ) -> tuple[np.ndarray, list[int]]:
        """Compute per-label per-family weights based on label distribution.

        Args:
            family_label_counts: List of [num_classes] counts per family.
            num_classes: Number of classes.

        Returns:
            label_weights: [num_classes, num_families] normalized weights.
            qualified_labels: Labels with sufficient samples.
        """
        num_families = len(family_label_counts)
        label_weights = np.zeros((num_classes, num_families))
        qualified_labels: list[int] = []

        for label in range(num_classes):
            weights = [counts[label] for counts in family_label_counts]
            total = sum(weights)

            if max(weights) > MIN_SAMPLES_PER_LABEL and total > 0:
                qualified_labels.append(label)
                label_weights[label] = np.array(weights) / total
            else:
                label_weights[label] = np.ones(num_families) / num_families

        if not qualified_labels:
            qualified_labels = list(range(num_classes))

        return label_weights, qualified_labels

    def train_generator(
        self,
        models: dict[str, nn.Module],
        label_weights: np.ndarray,
        qualified_labels: list[int],
        num_epochs: int = 1,
        num_teacher_iters: int = 20,
    ) -> float:
        """Train the generator using family models' forward_from_latent.

        Server-side objective:
            L = alpha * L_teacher + eta * L_diversity

        where:
        - L_teacher: Weighted CE of classifier(G(y)) across all families
        - L_diversity: Prevents mode collapse in generator

        After each optimizer step, the EMA tracker is updated. The EMA
        weights are exposed via get_inference_state_dict() for distribution
        to clients, providing smoother latent representations.

        Args:
            models: Dict mapping family name to FedGenModelWrapper.
            label_weights: [num_classes, num_families] per-label weights.
            qualified_labels: Labels with sufficient data.
            num_epochs: Generator training epochs per round.
            num_teacher_iters: Iterations per epoch.

        Returns:
            Average training loss.
        """
        if len(models) < 2:
            logger.info("Skipping generator training: need >= 2 models")
            return 0.0

        logger.info(
            f"Generator training: {len(models)} family models, "
            f"epochs={num_epochs}, iters={num_teacher_iters}"
        )

        self.generator.train()
        model_list = list(models.values())

        # Freeze all model parameters during generator training
        for model in model_list:
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

        total_loss = 0.0
        total_steps = 0

        for _epoch in range(num_epochs):
            for _step in range(num_teacher_iters):
                self.gen_optimizer.zero_grad()

                # Sample random labels from qualified labels
                y = np.random.choice(qualified_labels, self.batch_size)
                y_input = torch.LongTensor(y).to(self.device)

                # Generate latent vectors
                gen_result = self.generator(y_input)
                gen_output = gen_result["output"]
                eps = gen_result["eps"]

                # Teacher loss: weighted CE across all family models
                teacher_loss = torch.tensor(0.0, device=self.device)

                for model_idx, model in enumerate(model_list):
                    weight = label_weights[y, model_idx]
                    weight_tensor = torch.tensor(
                        weight, dtype=torch.float32, device=self.device
                    )

                    # Use forward_from_latent: classifier(z) only
                    logits = model.forward_from_latent(gen_output)
                    logp = F.log_softmax(logits, dim=1)

                    per_sample_loss = self.generator.crossentropy_loss(logp, y_input)
                    teacher_loss += torch.mean(per_sample_loss * weight_tensor)

                # Diversity loss
                diversity_loss = self.generator.diversity_loss(eps, gen_output)

                # Total loss
                loss = (
                    self.ensemble_alpha * teacher_loss
                    + self.ensemble_eta * diversity_loss
                )

                loss.backward()
                self.gen_optimizer.step()

                # Update EMA after each optimizer step
                if self.ema_tracker is not None:
                    self.ema_tracker.update(self.generator)

                total_loss += loss.item()
                total_steps += 1

        # Re-enable gradients for model parameters
        for model in model_list:
            for p in model.parameters():
                p.requires_grad = True

        avg_loss = total_loss / max(total_steps, 1)
        logger.info(f"Generator training done: avg_loss={avg_loss:.4f}")
        return avg_loss
