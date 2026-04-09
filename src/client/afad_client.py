"""AFAD client: HeteroFL shape-aware parameters + FedGen client-side KD.

Combines:
- HeteroFL: Shape-aware set_parameters for width-scaled sub-models
- FedGen: Client-side regularization via generator latent space
- AnchorKD (optional): Full-rate frozen anchor teacher for sub-rate clients

Training loss (base):
  L = CE(model(x), y)                                       # predictive
    + alpha * CE(classifier(G(y_rand)), y_rand)              # teacher
    + beta  * KL(model(x) || classifier(G(y_real)))          # latent matching

AnchorKD additions (sub-rate clients only, anchor_model != None):
  + gamma * T^2 * KL(student(x)/T || anchor(x)/T)           # logit-level KD
  + bn_gamma * MSE(LN(student_bn[:,:ed]), LN(anchor_bn[:,:ed]))  # bottleneck-level

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
        anchor_kd_gamma: Logit-level AnchorKD weight (sub-rate only, 0=disabled).
        bottleneck_gamma: Bottleneck-level MSE weight (sub-rate only, 0=disabled).
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
        proto_gamma: float = 0.0,
        anchor_kd_gamma: float = 0.0,
        bottleneck_gamma: float = 0.0,
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
        self.proto_gamma = proto_gamma
        self.anchor_kd_gamma = anchor_kd_gamma
        self.bottleneck_gamma = bottleneck_gamma
        # Anchor model for sub-rate KD (received from server each round)
        self.anchor_model: FedGenModelWrapper | None = None

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

    def set_anchor_parameters(self, parameters: list[np.ndarray]) -> None:
        """Set anchor model parameters (rate=1.0 family global model).

        The anchor model is a frozen copy of the full-rate family model used
        as a teacher for sub-rate clients in AnchorKD.
        """
        if self.anchor_model is None:
            # Lazily create anchor model with same architecture at rate=1.0
            from src.models.registry import ModelRegistry
            base = ModelRegistry.create_model(
                self.model_name, num_classes=self.num_classes, model_rate=1.0
            )
            self.anchor_model = FedGenModelWrapper(
                base, latent_dim=self.model.latent_dim, num_classes=self.num_classes
            )
            self.anchor_model.to(self.device)

        anchor_keys = list(self.anchor_model.state_dict().keys())
        state_dict = OrderedDict()
        for key, param in zip(anchor_keys, parameters):
            state_dict[key] = torch.tensor(param)
        self.anchor_model.load_state_dict(state_dict, strict=False)
        self.anchor_model.eval()
        for p in self.anchor_model.parameters():
            p.requires_grad = False

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

        # Update anchor model if params provided (sub-rate AnchorKD)
        anchor_params_bytes = config.get("anchor_params", None)
        if isinstance(anchor_params_bytes, bytes):
            anchor_params = pickle.loads(anchor_params_bytes)  # noqa: S301
            self.set_anchor_parameters(anchor_params)

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
            anchor_model=self.anchor_model,
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
        anchor_model: FedGenModelWrapper | None = None,
    ) -> None:
        """Local training with FedGen regularization and optional FedProx/AnchorKD.

        FedGen regularization (two terms from original paper):
        1. Teacher loss: CE on generated latents with random labels
        2. Latent matching: KL between real data output and generated output
        Both terms decay with 0.98^round, disabled after EARLY_STOP_EPOCH epochs.

        FedProx: proximal term (mu/2)||w - w_global||^2 for Non-IID stability.

        AnchorKD (sub-rate clients, anchor_model != None):
        - Logit-level: T^2 * KL(student(x)/T || anchor(x)/T), T=1/model_rate
        - Bottleneck-level: MSE(LN(student_bn[:,:ed]), LN(anchor_bn[:,:ed]))
          where ed = int(latent_dim * model_rate), LN = LayerNorm
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

                # AnchorKD: full-rate frozen model as teacher (sub-rate only)
                anchor_active = (
                    anchor_model is not None
                    and self.model_rate < 1.0
                    and (self.anchor_kd_gamma > 0.0 or self.bottleneck_gamma > 0.0)
                )
                if anchor_active:
                    T = 1.0 / self.model_rate  # temperature: 2.0 for rate=0.5, 4.0 for 0.25
                    with torch.no_grad():
                        anchor_logits = anchor_model(images)

                    # Logit-level KD: KL divergence with temperature scaling
                    if self.anchor_kd_gamma > 0.0:
                        anchor_p = F.softmax(anchor_logits / T, dim=1).detach()
                        student_logp = F.log_softmax(logits / T, dim=1)
                        anchor_kd_loss = (
                            self.anchor_kd_gamma
                            * T**2
                            * F.kl_div(student_logp, anchor_p, reduction="batchmean")
                        )
                        loss = loss + anchor_kd_loss

                    # Bottleneck-level KD: MSE on LayerNorm-aligned features
                    if self.bottleneck_gamma > 0.0:
                        latent_dim = self.model.latent_dim
                        ed = max(1, int(latent_dim * self.model_rate))
                        with torch.no_grad():
                            anchor_feats = anchor_model.backbone(images)
                            anchor_bn = anchor_model.bottleneck(anchor_feats)
                        student_feats = self.model.backbone(images)
                        student_bn = self.model.bottleneck(student_feats)
                        # LayerNorm before MSE to handle scale differences
                        s_norm = F.layer_norm(student_bn[:, :ed], [ed])
                        a_norm = F.layer_norm(anchor_bn[:, :ed].detach(), [ed])
                        bn_loss = self.bottleneck_gamma * F.mse_loss(s_norm, a_norm)
                        loss = loss + bn_loss

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
                        gen_result = self.generator(sampled_y_t, rate=self.model_rate)
                        gen_latent = gen_result["output"]

                    gen_logits = self.model.forward_from_latent(gen_latent)
                    gen_logp = F.log_softmax(gen_logits, dim=1)
                    teacher_loss = alpha * torch.mean(
                        FedGenGenerator.crossentropy_loss(gen_logp, sampled_y_t)
                    )

                    # Latent matching: KL(real output || generated output)
                    with torch.no_grad():
                        gen_result_same = self.generator(labels, rate=self.model_rate)
                        gen_latent_same = gen_result_same["output"]

                    gen_logits_same = self.model.forward_from_latent(gen_latent_same)
                    target_p = F.softmax(gen_logits_same, dim=1).clone().detach()
                    user_logp = F.log_softmax(logits, dim=1)
                    latent_loss = beta * F.kl_div(
                        user_logp, target_p, reduction="batchmean"
                    )

                    gen_ratio = self.gen_batch_size / images.size(0)
                    loss = loss + gen_ratio * teacher_loss + latent_loss

                    # Prototype anchoring: directly align bottleneck output
                    # with generator latents at the geometric level.
                    # Minimizes MSE(bottleneck(backbone(x)), G(y)) so that
                    # sub-rate clients' 32-dim latents stay close to the
                    # shared space that the generator was trained on.
                    if self.proto_gamma > 0.0:
                        gamma = self.proto_gamma * (DECAY_RATE**glob_iter)
                        with torch.no_grad():
                            proto_target = self.generator(labels, rate=self.model_rate)["output"]
                        client_latent = self.model.bottleneck(
                            self.model.backbone(images)
                        )
                        proto_loss = gamma * F.mse_loss(client_latent, proto_target)
                        loss = loss + proto_loss

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
