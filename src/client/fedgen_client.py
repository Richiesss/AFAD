"""Faithful FedGen client (Zhu et al., ICML 2021).

Client-side training with generator-based regularization:
  L = CE(model(x), y)                          # predictive loss
    + α × CE(model.from_latent(G(y_rand)), y_rand)  # teacher regularization
    + β × KL(model(x) || model.from_latent(G(y_real)))  # latent matching

where α=10, β=10 with 0.98/round exponential decay.
"""

from collections import OrderedDict

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.fedgen_wrapper import FedGenModelWrapper
from src.server.generator.fedgen_generator import FedGenGenerator
from src.utils.logger import setup_logger

logger = setup_logger("FedGenClient")

# Default hyperparameters matching the original paper
DEFAULT_GENERATIVE_ALPHA = 10.0
DEFAULT_GENERATIVE_BETA = 10.0
DECAY_RATE = 0.98
EARLY_STOP_EPOCH = 20


def exp_lr_decay(glob_iter: int, init_lr: float, decay: float = DECAY_RATE) -> float:
    """Exponential learning rate decay: init_lr * decay^glob_iter."""
    return init_lr * (decay**glob_iter)


class FedGenClient(fl.client.NumPyClient):
    """FedGen client with client-side knowledge distillation.

    Differs from AFADClient in that KD happens during local training
    (client-side) rather than on the server. The generator produces
    latent representations that regularize the client's training.

    Args:
        cid: Client identifier.
        model: FedGenModelWrapper with latent-layer support.
        generator: FedGenGenerator (shared, received from server).
        train_loader: Training data loader.
        epochs: Number of local training epochs.
        device: Compute device.
        lr: Learning rate.
        weight_decay: Weight decay for optimizer.
        generative_alpha: Teacher loss weight (default 10.0).
        generative_beta: Latent matching loss weight (default 10.0).
        gen_batch_size: Batch size for generated samples.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        cid: str,
        model: FedGenModelWrapper,
        generator: FedGenGenerator,
        train_loader,
        val_loader=None,
        epochs: int = 20,
        device: str = "cpu",
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-2,
        generative_alpha: float = DEFAULT_GENERATIVE_ALPHA,
        generative_beta: float = DEFAULT_GENERATIVE_BETA,
        gen_batch_size: int = 32,
        num_classes: int = 10,
        family: str = "default",
    ):
        self.cid = cid
        self.model = model
        self.generator = generator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = torch.device(device)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.family = family
        self.generative_alpha = generative_alpha
        self.generative_beta = generative_beta
        self.gen_batch_size = gen_batch_size
        self.num_classes = num_classes

        self.model.to(self.device)
        self.generator.to(self.device)

        # Precompute label counts and available labels
        self.label_counts = self._compute_label_counts()
        self.available_labels = [i for i, c in enumerate(self.label_counts) if c > 0]

        logger.info(f"FedGenClient {cid} initialized: device={self.device}")

    def _compute_label_counts(self) -> list[int]:
        """Count per-class samples in training data."""
        counts = [0] * self.num_classes
        for _, labels in self.train_loader:
            for label in labels:
                idx = label.item()
                if idx < self.num_classes:
                    counts[idx] += 1
        return counts

    def get_parameters(self, config) -> list[np.ndarray]:
        """Return model parameters as NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Set model parameters from NumPy arrays."""
        model_keys = list(self.model.state_dict().keys())
        state_dict = OrderedDict()
        for key, param in zip(model_keys, parameters):
            state_dict[key] = torch.tensor(param)
        self.model.load_state_dict(state_dict, strict=False)

    def set_generator_parameters(self, parameters: list[np.ndarray]) -> None:
        """Set generator parameters from NumPy arrays."""
        gen_keys = list(self.generator.state_dict().keys())
        state_dict = OrderedDict()
        for key, param in zip(gen_keys, parameters):
            state_dict[key] = torch.tensor(param)
        self.generator.load_state_dict(state_dict, strict=False)

    def fit(
        self, parameters: list[np.ndarray], config: dict
    ) -> tuple[list[np.ndarray], int, dict]:
        """Train the model with FedGen client-side regularization."""
        if parameters and len(parameters) > 0:
            self.set_parameters(parameters)

        # Propagate training config from server (mirrors AFADClient behaviour)
        if "family" in config:
            self.family = config["family"]
        if "lr" in config:
            self.lr = config["lr"]
        if "momentum" in config:
            self.momentum = config["momentum"]
        if "weight_decay" in config:
            self.weight_decay = config["weight_decay"]
        if "local_epochs" in config:
            self.epochs = config["local_epochs"]

        glob_iter = config.get("round", 0)
        regularization = config.get("regularization", glob_iter > 0)

        # Update generator if params provided in config
        gen_params_bytes = config.get("generator_params", None)
        if isinstance(gen_params_bytes, bytes):
            import pickle

            gen_params = pickle.loads(gen_params_bytes)  # noqa: S301
            self.set_generator_parameters(gen_params)

        self._train(glob_iter=glob_iter, regularization=regularization)

        label_counts_str = ",".join(str(c) for c in self.label_counts)

        return (
            self.get_parameters(config={}),
            len(self.train_loader.dataset),
            {
                "client_id": self.cid,
                "family": getattr(self, "family", "default"),
                "label_counts": label_counts_str,
            },
        )

    def _train(self, glob_iter: int = 0, regularization: bool = True) -> None:
        """Local training with FedGen client-side regularization.

        Two regularization terms (from original paper):
        1. Teacher loss: CE on generated latents with random labels
        2. Latent matching: KL between real data output and generated output

        Both terms decay with 0.98^round and are disabled after EARLY_STOP_EPOCH
        local epochs.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.model.train()
        self.generator.eval()

        for epoch in range(self.epochs):
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()

                # Standard predictive loss on real data
                logits = self.model(images)
                predictive_loss = criterion(logits, labels)

                # FedGen regularization (after first round, before early_stop)
                if regularization and epoch < EARLY_STOP_EPOCH:
                    alpha = exp_lr_decay(glob_iter, self.generative_alpha)
                    beta = exp_lr_decay(glob_iter, self.generative_beta)

                    # === Teacher loss (α term) ===
                    # Generate latents for RANDOM labels
                    sampled_y = np.random.choice(
                        self.available_labels, self.gen_batch_size
                    )
                    sampled_y_tensor = torch.tensor(
                        sampled_y, dtype=torch.long, device=self.device
                    )
                    with torch.no_grad():
                        gen_result = self.generator(sampled_y_tensor)
                        gen_latent = gen_result["output"]

                    gen_logits = self.model.forward_from_latent(gen_latent)
                    gen_logp = F.log_softmax(gen_logits, dim=1)
                    teacher_loss = alpha * torch.mean(
                        FedGenGenerator.crossentropy_loss(gen_logp, sampled_y_tensor)
                    )

                    # === Latent matching loss (β term) ===
                    # Generate latents for the SAME labels as real batch
                    with torch.no_grad():
                        gen_result_same = self.generator(labels)
                        gen_latent_same = gen_result_same["output"]

                    gen_logits_same = self.model.forward_from_latent(gen_latent_same)
                    target_p = F.softmax(gen_logits_same, dim=1).clone().detach()

                    user_output_logp = F.log_softmax(logits, dim=1)
                    latent_loss = beta * F.kl_div(
                        user_output_logp, target_p, reduction="batchmean"
                    )

                    # Total loss with gen_ratio scaling (original paper)
                    gen_ratio = self.gen_batch_size / images.size(0)
                    loss = predictive_loss + gen_ratio * teacher_loss + latent_loss
                else:
                    loss = predictive_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                optimizer.step()

    def evaluate(
        self, parameters: list[np.ndarray], config: dict
    ) -> tuple[float, int, dict]:
        """Evaluate the model on validation or training data."""
        if parameters:
            self.set_parameters(parameters)

        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        eval_loader = self.val_loader if self.val_loader is not None else self.train_loader
        with torch.no_grad():
            for images, labels in eval_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, total, {"accuracy": accuracy}
