import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.server.generator.synthetic_generator import SyntheticGenerator
from src.utils.logger import setup_logger

logger = setup_logger("FedGenDistiller")

# Minimum samples per label to include in generator training
MIN_SAMPLES_PER_LABEL = 1


class FedGenDistiller:
    """
    FedGen server-side generator training and knowledge distillation.

    Two-phase operation per round:
    1. Train generator to produce synthetic data that minimizes ensemble disagreement
    2. Distill knowledge between models using generated data and KD loss

    Reference: Zhu et al., "Data-Free Knowledge Distillation for Heterogeneous
    Federated Learning" (ICML 2021)

    Adapted for heterogeneous architectures: generator produces images (not
    latent representations) so any model architecture can consume them.
    Knowledge distillation happens server-side (not client-side) to avoid
    contaminating client training with low-quality generated images.
    """

    def __init__(
        self,
        generator: SyntheticGenerator,
        gen_lr: float = 3e-4,
        batch_size: int = 128,
        ensemble_alpha: float = 1.0,
        ensemble_eta: float = 1.0,
        device: str = "cpu",
        temperature: float = 4.0,
        distill_lr: float = 1e-4,
        distill_epochs: int = 1,
        distill_steps: int = 5,
        distill_alpha: float = 1.0,
        distill_beta: float = 0.1,
        distill_every: int = 2,
    ):
        self.generator = generator
        self.gen_lr = gen_lr
        self.batch_size = batch_size
        self.ensemble_alpha = ensemble_alpha  # Teacher loss weight
        self.ensemble_eta = ensemble_eta  # Diversity loss weight
        self.device = device
        self.temperature = temperature
        self.distill_lr = distill_lr
        self.distill_epochs = distill_epochs
        self.distill_steps = distill_steps
        self.distill_alpha = distill_alpha
        self.distill_beta = distill_beta  # EMA blending factor (0.1 = 10% distilled)
        self.distill_every = distill_every  # Distill every N rounds after warmup
        self.generator.to(device)
        self.gen_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=self.gen_lr
        )

    def get_label_weights(
        self,
        client_label_counts: list[list[int]],
        num_classes: int = 10,
    ) -> tuple[np.ndarray, list[int]]:
        """
        Compute per-label per-client weights based on label distribution.

        Args:
            client_label_counts: List of [num_classes] counts per client
            num_classes: Number of classes

        Returns:
            label_weights: [num_classes, num_clients] normalized weights
            qualified_labels: Labels with sufficient samples
        """
        num_clients = len(client_label_counts)
        label_weights = np.zeros((num_classes, num_clients))
        qualified_labels = []

        for label in range(num_classes):
            weights = [counts[label] for counts in client_label_counts]
            total = sum(weights)

            if max(weights) > MIN_SAMPLES_PER_LABEL and total > 0:
                qualified_labels.append(label)
                label_weights[label] = np.array(weights) / total
            else:
                # Uniform fallback
                label_weights[label] = np.ones(num_clients) / num_clients

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
        """
        Train the generator using the ensemble of client models.

        Server-side objective:
            L = α * L_teacher + η * L_diversity

        where:
        - L_teacher: Weighted CE across all client models
        - L_diversity: Prevents mode collapse in generator

        Args:
            models: Dict mapping signature to nn.Module (client models)
            label_weights: [num_classes, num_models] per-label weights
            qualified_labels: Labels with sufficient data
            num_epochs: Generator training epochs per round
            num_teacher_iters: Iterations per epoch

        Returns:
            Average training loss
        """
        if len(models) < 2:
            logger.info("Skipping generator training: need >= 2 models")
            return 0.0

        logger.info(
            f"Generator training: {len(models)} models, "
            f"epochs={num_epochs}, iters={num_teacher_iters}"
        )

        self.generator.train()
        model_list = list(models.values())

        # Freeze all model parameters
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

                # Generate synthetic data
                gen_result = self.generator(y_input)
                gen_output = gen_result["output"]
                eps = gen_result["eps"]

                # === Teacher Loss: Weighted CE across all models ===
                teacher_loss = torch.tensor(0.0, device=self.device)

                for model_idx, model in enumerate(model_list):
                    # Per-sample weights based on label distribution
                    weight = label_weights[y, model_idx]
                    weight_tensor = torch.tensor(
                        weight, dtype=torch.float32, device=self.device
                    )

                    # Forward through model
                    logits = model(gen_output)
                    logp = F.log_softmax(logits, dim=1)

                    # Weighted per-sample CE
                    per_sample_loss = self.generator.crossentropy_loss(logp, y_input)
                    teacher_loss += torch.mean(per_sample_loss * weight_tensor)

                # === Diversity Loss ===
                diversity_loss = self.generator.diversity_loss(eps, gen_output)

                # === Total Loss ===
                loss = (
                    self.ensemble_alpha * teacher_loss
                    + self.ensemble_eta * diversity_loss
                )

                loss.backward()
                self.gen_optimizer.step()

                total_loss += loss.item()
                total_steps += 1

        # Re-enable gradients for model parameters
        for model in model_list:
            for p in model.parameters():
                p.requires_grad = True

        avg_loss = total_loss / max(total_steps, 1)
        logger.info(f"Generator training done: avg_loss={avg_loss:.4f}")
        return avg_loss

    def _check_generator_quality(
        self,
        models: dict[str, nn.Module],
        qualified_labels: list[int],
        num_samples: int = 128,
    ) -> float:
        """
        Check generator quality by measuring ensemble accuracy on generated data.

        Returns the fraction of generated samples where the ensemble's
        top-1 prediction matches the conditioned label.
        """
        self.generator.eval()
        y = np.random.choice(qualified_labels, num_samples)
        y_input = torch.LongTensor(y).to(self.device)

        with torch.no_grad():
            gen_result = self.generator(y_input)
            gen_images = gen_result["output"]

            # Ensemble prediction: average logits from all models
            ensemble_logits = torch.zeros(
                num_samples, max(qualified_labels) + 1, device=self.device
            )
            for model in models.values():
                model.eval()
                ensemble_logits += model(gen_images)
            ensemble_logits /= len(models)

            preds = ensemble_logits.argmax(dim=1)
            accuracy = (preds == y_input).float().mean().item()

        return accuracy

    def distill_models(
        self,
        models: dict[str, nn.Module],
        label_weights: np.ndarray,
        qualified_labels: list[int],
        num_epochs: int | None = None,
        num_steps: int | None = None,
        temperature: float | None = None,
        distill_alpha: float | None = None,
        distill_lr: float | None = None,
        min_quality: float = 0.4,
    ) -> dict[str, float]:
        """
        Server-side knowledge distillation between models via generator.

        For each model (student), uses all OTHER models as ensemble teachers.
        Fine-tunes the student with KD loss on generated pseudo-data.
        Includes a quality gate: skips distillation if the ensemble cannot
        classify generated images above min_quality accuracy.

        Args:
            models: Dict mapping signature to nn.Module
            label_weights: [num_classes, num_models] per-label weights
            qualified_labels: Labels with sufficient data
            num_epochs: Distillation epochs (default: self.distill_epochs)
            num_steps: Gradient steps per epoch (default: self.distill_steps)
            temperature: KD temperature (default: self.temperature)
            distill_alpha: KD loss weight (default: self.distill_alpha)
            distill_lr: Learning rate (default: self.distill_lr)
            min_quality: Minimum ensemble accuracy on generated data to proceed

        Returns:
            Dict mapping signature to average distillation loss
        """
        if len(models) < 2:
            logger.info("Skipping distillation: need >= 2 models")
            return {}

        num_epochs = num_epochs if num_epochs is not None else self.distill_epochs
        num_steps = num_steps if num_steps is not None else self.distill_steps
        temperature = temperature if temperature is not None else self.temperature
        alpha = distill_alpha if distill_alpha is not None else self.distill_alpha
        lr = distill_lr if distill_lr is not None else self.distill_lr
        beta = self.distill_beta

        logger.info(
            f"Distillation: {len(models)} models, "
            f"epochs={num_epochs}, steps={num_steps}, T={temperature}, "
            f"α={alpha}, β={beta}"
        )

        # Freeze generator
        self.generator.eval()
        for p in self.generator.parameters():
            p.requires_grad = False

        # Quality gate: check if ensemble can classify generated images
        gen_quality = self._check_generator_quality(models, qualified_labels)
        if gen_quality < min_quality:
            logger.info(
                f"Skipping distillation: generator quality too low "
                f"(ensemble_acc={gen_quality:.2%}, threshold=40%)"
            )
            for p in self.generator.parameters():
                p.requires_grad = True
            return {}

        logger.info(f"Generator quality check passed: ensemble_acc={gen_quality:.2%}")

        model_sigs = list(models.keys())
        distill_losses: dict[str, float] = {}

        for student_sig in model_sigs:
            student = models[student_sig]

            # Save original weights before distillation (for EMA blending)
            original_state = {
                name: param.data.clone() for name, param in student.named_parameters()
            }

            # Use eval mode to preserve BN running stats from real data
            student.eval()
            for p in student.parameters():
                p.requires_grad = True

            # Freeze all teacher models
            teachers = {sig: m for sig, m in models.items() if sig != student_sig}
            for teacher in teachers.values():
                teacher.eval()
                for p in teacher.parameters():
                    p.requires_grad = False

            optimizer = torch.optim.Adam(student.parameters(), lr=lr)

            total_loss = 0.0
            total_steps_done = 0

            for _epoch in range(num_epochs):
                for _step in range(num_steps):
                    optimizer.zero_grad()

                    # Sample labels and generate pseudo-data
                    y = np.random.choice(qualified_labels, self.batch_size)
                    y_input = torch.LongTensor(y).to(self.device)

                    with torch.no_grad():
                        gen_result = self.generator(y_input)
                        gen_images = gen_result["output"]

                    # Compute ensemble soft labels from teachers
                    with torch.no_grad():
                        soft_targets = torch.zeros(
                            self.batch_size,
                            label_weights.shape[0],
                            device=self.device,
                        )
                        num_teachers = 0
                        for teacher in teachers.values():
                            logits = teacher(gen_images)
                            soft_targets += F.softmax(logits / temperature, dim=1)
                            num_teachers += 1
                        soft_targets /= num_teachers

                    # Student forward on generated images
                    student_logits = student(gen_images)
                    student_log_soft = F.log_softmax(
                        student_logits / temperature, dim=1
                    )

                    # KD loss with T² scaling (Hinton et al., 2015):
                    # Compensates for 1/T² factor in softmax gradients
                    kd_loss = temperature**2 * F.kl_div(
                        student_log_soft, soft_targets, reduction="batchmean"
                    )

                    if alpha < 1.0:
                        # Hard label loss (CE on generated images)
                        hard_loss = F.cross_entropy(student_logits, y_input)
                        loss = alpha * kd_loss + (1 - alpha) * hard_loss
                    else:
                        loss = kd_loss

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total_steps_done += 1

            # EMA blending: new = (1-β)*original + β*distilled
            # Keeps most of the real-data-trained weights, gently absorbs KD
            with torch.no_grad():
                for name, param in student.named_parameters():
                    param.data = (1 - beta) * original_state[name] + beta * param.data

            avg_loss = total_loss / max(total_steps_done, 1)
            distill_losses[student_sig] = avg_loss

            # Re-enable teacher gradients
            for teacher in teachers.values():
                for p in teacher.parameters():
                    p.requires_grad = True

        # Re-enable generator gradients
        for p in self.generator.parameters():
            p.requires_grad = True

        logger.info(
            "Distillation done: "
            + ", ".join(f"avg_loss={v:.4f}" for v in distill_losses.values())
        )
        return distill_losses
