import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import setup_logger

logger = setup_logger("FedGenDistiller")


class FedGenDistiller:
    """
    FedGen-style inter-family knowledge distillation.

    Two phases:
    1. Generator Training: Train generator so synthetic data produces
       consistent logits across the model ensemble.
    2. Model Distillation: Use KD loss (KL divergence with temperature)
       to transfer ensemble knowledge to each individual model.
    """

    def __init__(
        self,
        generator: nn.Module,
        temperature: float = 4.0,
        device: str = "cpu",
        gen_lr: float = 0.001,
        distill_lr: float = 0.001,
        batch_size: int = 32,
    ):
        self.generator = generator
        self.temperature = temperature
        self.device = device
        self.batch_size = batch_size
        self.gen_lr = gen_lr
        self.distill_lr = distill_lr
        self.generator.to(device)
        self.gen_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=self.gen_lr
        )

    def distill(
        self,
        models: dict[str, nn.Module],
        gen_steps: int = 10,
        distill_steps: int = 5,
    ) -> dict[str, nn.Module]:
        """
        Run full FedGen distillation pipeline.

        Args:
            models: dict mapping model name/signature to nn.Module
            gen_steps: number of generator training steps
            distill_steps: number of distillation steps per model

        Returns:
            Updated models dict with distilled knowledge
        """
        if len(models) < 2:
            logger.info("Skipping FedGen: need >= 2 models for cross-distillation")
            return models

        logger.info(
            f"FedGen distillation: {len(models)} models, "
            f"gen_steps={gen_steps}, distill_steps={distill_steps}"
        )

        # Move all models to device
        for model in models.values():
            model.to(self.device)

        # Phase 1: Train generator
        self._train_generator(models, gen_steps)
        self.generator.update_ema()

        # Phase 2: Distill ensemble knowledge to each model
        self._distill_to_models(models, distill_steps)

        return models

    def _train_generator(self, models: dict[str, nn.Module], num_steps: int) -> None:
        """
        Phase 1: Train generator to produce images that the ensemble
        classifies consistently as the target labels.
        """
        self.generator.train()

        # Freeze all model parameters
        for model in models.values():
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

        total_loss = 0.0
        for step in range(num_steps):
            self.gen_optimizer.zero_grad()

            # Generate batch (normalization applied inside generate_batch)
            images, labels = self.generator.generate_batch(
                self.batch_size, device=self.device
            )

            # Compute ensemble logits (average across all models)
            ensemble_logits = self._compute_ensemble_logits(models, images)

            # Generator loss: encourage ensemble to classify correctly
            loss = F.cross_entropy(ensemble_logits, labels)
            loss.backward()
            self.gen_optimizer.step()
            total_loss += loss.item()

        # Re-enable gradient computation for model parameters
        for model in models.values():
            for p in model.parameters():
                p.requires_grad = True

        avg_loss = total_loss / max(num_steps, 1)
        logger.info(f"Generator training done: avg_loss={avg_loss:.4f}")

    def _distill_to_models(self, models: dict[str, nn.Module], num_steps: int) -> None:
        """
        Phase 2: Distill ensemble knowledge to each individual model
        using KL divergence with temperature scaling.
        """
        T = self.temperature

        for name, student in models.items():
            # Create a separate optimizer for each student
            student_optimizer = torch.optim.SGD(
                student.parameters(), lr=self.distill_lr, momentum=0.9
            )
            student.train()

            total_loss = 0.0
            for step in range(num_steps):
                # Generate synthetic data
                with torch.no_grad():
                    images, labels = self.generator.generate_batch(
                        self.batch_size, device=self.device, use_ema=True
                    )

                # Compute ensemble teacher logits (exclude current student)
                with torch.no_grad():
                    teacher_logits = self._compute_ensemble_logits(
                        {k: v for k, v in models.items() if k != name},
                        images,
                    )

                # Student forward pass
                student_logits = student(images)

                # KD loss: KL divergence with temperature
                loss = self._kd_loss(student_logits, teacher_logits, T)

                student_optimizer.zero_grad()
                loss.backward()
                student_optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(num_steps, 1)
            logger.info(f"Distilled model '{name}': avg_kd_loss={avg_loss:.4f}")

    def _compute_ensemble_logits(
        self, models: dict[str, nn.Module], images: torch.Tensor
    ) -> torch.Tensor:
        """Compute average logits across all models in the ensemble."""
        logits_sum = None
        for model in models.values():
            out = model(images)
            if logits_sum is None:
                logits_sum = out
            else:
                logits_sum = logits_sum + out
        return logits_sum / len(models)

    @staticmethod
    def _kd_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """
        Knowledge distillation loss using KL divergence with temperature.

        KD_loss = T^2 * KL(softmax(student/T) || softmax(teacher/T))
        """
        student_soft = F.log_softmax(student_logits / temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
        return F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (
            temperature**2
        )
