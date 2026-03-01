"""AFAD client: HeteroFL shape-aware parameters + FedGen client-side KD.

Combines:
- HeteroFL: Shape-aware set_parameters for width-scaled sub-models
- FedGen: Client-side regularization via generator latent space

Training loss:
  L = CE(model(x), y)                                       # predictive
    + alpha * CE(classifier(G(y_rand)), y_rand)              # teacher
    + beta  * KL(model(x) || classifier(G(y_real)))          # latent matching

where alpha=10, beta=10, decaying by 0.98/round.
"""

import pickle
from collections import OrderedDict

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.fedgen_wrapper import FedGenModelWrapper
from src.server.generator.fedgen_generator import FedGenGenerator
from src.utils.logger import setup_logger

logger = setup_logger("AFADClient")

# FedGen hyperparameters (Zhu et al., ICML 2021)
DEFAULT_GENERATIVE_ALPHA = 10.0
DEFAULT_GENERATIVE_BETA = 10.0
DECAY_RATE = 0.98
EARLY_STOP_EPOCH = 20


class AFADClient(fl.client.NumPyClient):
    """AFAD hybrid client: HeteroFL sub-model support + FedGen regularization.

    Key differences from HeteroFLClient:
    - Model is FedGenModelWrapper (has forward_from_latent)
    - Training includes FedGen regularization (teacher + latent matching)
    - Receives generator params via config["generator_params"]

    Key differences from FedGenClient:
    - Shape-aware set_parameters for HeteroFL width-scaled sub-models
    - FedProx proximal term support
    - Training config propagation (lr, momentum, weight_decay from server)

    Args:
        cid: Client identifier.
        model: FedGenModelWrapper with latent-layer support.
        generator: FedGenGenerator (shared, received from server).
        train_loader: Training data loader.
        epochs: Number of local training epochs.
        device: Compute device.
        family: Model family name (e.g., "cnn", "vit").
        model_rate: HeteroFL width rate (1.0 = full, 0.5 = half).
        model_name: Registry model name.
        val_loader: Validation data loader (optional).
        lr: Learning rate.
        momentum: SGD momentum.
        weight_decay: Weight decay.
        num_classes: Number of output classes.
        generative_alpha: Teacher loss weight.
        generative_beta: Latent matching loss weight.
        gen_batch_size: Batch size for generated samples.
    """

    def __init__(
        self,
        cid: str,
        model: FedGenModelWrapper,
        generator: FedGenGenerator,
        train_loader,
        epochs: int = 5,
        device: str = "cpu",
        family: str = "default",
        model_rate: float = 1.0,
        model_name: str = "",
        val_loader=None,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        num_classes: int = 10,
        generative_alpha: float = DEFAULT_GENERATIVE_ALPHA,
        generative_beta: float = DEFAULT_GENERATIVE_BETA,
        gen_batch_size: int = 32,
    ):
        self.cid = cid
        self.model = model
        self.generator = generator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = torch.device(device)
        self.model.to(self.device)
        self.generator.to(self.device)
        self.family = family
        self.model_rate = model_rate
        self.model_name = model_name
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.generative_alpha = generative_alpha
        self.generative_beta = generative_beta
        self.gen_batch_size = gen_batch_size

        # Precompute label counts and available labels
        self.label_counts = self._compute_label_counts()
        self.available_labels = [i for i, c in enumerate(self.label_counts) if c > 0]

        logger.info(
            f"Client {cid} initialized: model={model_name}, device={self.device}, "
            f"family={family}, model_rate={model_rate}"
        )

    def _compute_label_counts(self) -> list[int]:
        """Count per-class samples in training data (computed once)."""
        counts = [0] * self.num_classes
        for _, labels in self.train_loader:
            for label in labels:
                idx = label.item()
                if idx < self.num_classes:
                    counts[idx] += 1
        return counts

    def get_parameters(self, config) -> list[np.ndarray]:
        """Return model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: list[np.ndarray], config: dict = None):
        """Set model parameters with shape-aware handling for HeteroFL sub-models.

        Handles shape mismatches between distributed sub-model params and
        local model params by copying into the leading portion of each tensor.
        """
        model_keys = list(self.model.state_dict().keys())

        if len(parameters) != len(model_keys):
            logger.warning(
                f"Client {self.cid}: Parameter count mismatch. "
                f"Received {len(parameters)}, model has {len(model_keys)}"
            )

        state_dict = OrderedDict()
        for key, param in zip(model_keys, parameters):
            local_shape = self.model.state_dict()[key].shape
            received_shape = param.shape

            if local_shape == received_shape:
                state_dict[key] = torch.tensor(param)
            else:
                logger.debug(
                    f"Client {self.cid}: Layer {key} shape mismatch. "
                    f"Local: {local_shape}, Received: {received_shape}"
                )
                new_param = torch.zeros(local_shape)
                slices = tuple(
                    slice(0, min(loc, rec))
                    for loc, rec in zip(local_shape, received_shape)
                )
                new_param[slices] = torch.tensor(param)[slices]

                state_dict[key] = new_param

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
        """Train the model with HeteroFL params + FedGen regularization."""
        if "model_rate" in config:
            self.model_rate = config["model_rate"]
        if "family" in config:
            self.family = config["family"]

        # Propagate training config from server
        if "lr" in config:
            self.lr = config["lr"]
        if "momentum" in config:
            self.momentum = config["momentum"]
        if "weight_decay" in config:
            self.weight_decay = config["weight_decay"]
        if "local_epochs" in config:
            self.epochs = config["local_epochs"]

        fedprox_mu = config.get("fedprox_mu", 0.0)

        # Set model parameters if provided (shape-aware for HeteroFL)
        use_local_init = config.get("use_local_init", False)
        if parameters and len(parameters) > 0 and not use_local_init:
            self.set_parameters(parameters, config)

        # Update generator if params provided in config
        gen_params_bytes = config.get("generator_params", None)
        if isinstance(gen_params_bytes, bytes):
            gen_params = pickle.loads(gen_params_bytes)  # noqa: S301
            self.set_generator_parameters(gen_params)

        glob_iter = config.get("round", 0)
        regularization = config.get("regularization", glob_iter > 0)

        # Snapshot global params for FedProx proximal term
        global_params = (
            [p.clone().detach() for p in self.model.parameters()]
            if fedprox_mu > 0
            else None
        )

        self._train(
            glob_iter=glob_iter,
            regularization=regularization,
            fedprox_mu=fedprox_mu,
            global_params=global_params,
        )

        # Serialize label_counts as comma-separated string for Flower Scalar
        label_counts_str = ",".join(str(c) for c in self.label_counts)

        return (
            self.get_parameters(config={}),
            len(self.train_loader.dataset),
            {
                "family": self.family,
                "model_rate": self.model_rate,
                "client_id": self.cid,
                "model_name": self.model_name,
                "label_counts": label_counts_str,
            },
        )

    def _train(
        self,
        glob_iter: int = 0,
        regularization: bool = True,
        fedprox_mu: float = 0.0,
        global_params: list[torch.Tensor] | None = None,
    ) -> None:
        """Local training with FedGen regularization and optional FedProx.

        FedGen regularization (two terms from original paper):
        1. Teacher loss: CE on generated latents with random labels
        2. Latent matching: KL between real data output and generated output
        Both terms decay with 0.98^round, disabled after EARLY_STOP_EPOCH epochs.

        FedProx: proximal term (mu/2)||w - w_global||^2 for Non-IID stability.
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
                loss = criterion(logits, labels)

                # FedProx proximal term
                if fedprox_mu > 0 and global_params is not None:
                    prox_term = torch.tensor(0.0, device=self.device)
                    for local_p, global_p in zip(
                        self.model.parameters(), global_params
                    ):
                        prox_term += (local_p - global_p.to(self.device)).pow(2).sum()
                    loss = loss + (fedprox_mu / 2.0) * prox_term

                # FedGen regularization (after first round, before early stop)
                if regularization and epoch < EARLY_STOP_EPOCH:
                    alpha = self.generative_alpha * (DECAY_RATE**glob_iter)
                    beta = self.generative_beta * (DECAY_RATE**glob_iter)

                    # Teacher loss: CE on generated latents with random labels
                    sampled_y = np.random.choice(
                        self.available_labels, self.gen_batch_size
                    )
                    sampled_y_t = torch.tensor(
                        sampled_y, dtype=torch.long, device=self.device
                    )
                    with torch.no_grad():
                        gen_result = self.generator(sampled_y_t)
                        gen_latent = gen_result["output"]

                    gen_logits = self.model.forward_from_latent(gen_latent)
                    gen_logp = F.log_softmax(gen_logits, dim=1)
                    teacher_loss = alpha * torch.mean(
                        FedGenGenerator.crossentropy_loss(gen_logp, sampled_y_t)
                    )

                    # Latent matching: KL(real output || generated output)
                    with torch.no_grad():
                        gen_result_same = self.generator(labels)
                        gen_latent_same = gen_result_same["output"]

                    gen_logits_same = self.model.forward_from_latent(gen_latent_same)
                    target_p = F.softmax(gen_logits_same, dim=1).clone().detach()
                    user_logp = F.log_softmax(logits, dim=1)
                    latent_loss = beta * F.kl_div(
                        user_logp, target_p, reduction="batchmean"
                    )

                    gen_ratio = self.gen_batch_size / images.size(0)
                    loss = loss + gen_ratio * teacher_loss + latent_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                optimizer.step()

    def evaluate(
        self, parameters: list[np.ndarray], config: dict
    ) -> tuple[float, int, dict]:
        """Evaluate the model on validation data (or train data as fallback)."""
        if parameters:
            self.set_parameters(parameters, config)

        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        eval_loader = (
            self.val_loader if self.val_loader is not None else self.train_loader
        )

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in eval_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return (
            avg_loss,
            total,
            {
                "accuracy": accuracy,
                "family": self.family,
                "model_rate": self.model_rate,
                "model_name": self.model_name,
            },
        )
