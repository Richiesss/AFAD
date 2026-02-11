from collections import OrderedDict

import flwr as fl
import numpy as np
import torch
import torch.nn as nn

from src.utils.logger import setup_logger

logger = setup_logger("AFADClient")


class AFADClient(fl.client.NumPyClient):
    """
    AFAD Client with HeteroFL support.

    Standard local training with SGD. Knowledge distillation is handled
    server-side (not client-side) to avoid contaminating training with
    low-quality generated images.
    """

    def __init__(
        self,
        cid: str,
        model: nn.Module,
        train_loader,
        epochs: int,
        device: str = "cpu",
        family: str = "default",
        model_rate: float = 1.0,
        model_name: str = "",
        val_loader=None,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        num_classes: int = 10,
    ):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = torch.device(device)
        self.model.to(self.device)
        self.family = family
        self.model_rate = model_rate
        self.model_name = model_name
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.num_classes = num_classes

        # Precompute label counts from training data
        self.label_counts = self._compute_label_counts()

        logger.info(
            f"Client {cid} initialized: model={model_name}, device={self.device}, "
            f"family={family}, model_rate={model_rate}"
        )

    def _compute_label_counts(self) -> list[int]:
        """Count per-class samples in training data (computed once)."""
        counts = [0] * self.num_classes
        for _, labels in self.train_loader:
            for label in labels:
                counts[label.item()] += 1
        return counts

    def get_parameters(self, config) -> list[np.ndarray]:
        """Return model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: list[np.ndarray], config: dict = None):
        """
        Set model parameters from a list of NumPy arrays.

        Handles HeteroFL sub-models where parameters may have different shapes.
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
                if len(local_shape) == 1:
                    new_param[: received_shape[0]] = torch.tensor(param)
                elif len(local_shape) == 2:
                    new_param[: received_shape[0], : received_shape[1]] = torch.tensor(
                        param
                    )
                elif len(local_shape) == 4:
                    new_param[: received_shape[0], : received_shape[1], :, :] = (
                        torch.tensor(param)
                    )
                else:
                    slices = tuple(
                        slice(0, min(loc, rec))
                        for loc, rec in zip(local_shape, received_shape)
                    )
                    new_param[slices] = torch.tensor(param[slices])

                state_dict[key] = new_param

        self.model.load_state_dict(state_dict, strict=False)

    def fit(
        self, parameters: list[np.ndarray], config: dict
    ) -> tuple[list[np.ndarray], int, dict]:
        """Train the model on local data."""
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

        # Set model parameters if provided
        use_local_init = config.get("use_local_init", False)
        if parameters and len(parameters) > 0 and not use_local_init:
            self.set_parameters(parameters, config)

        self._train()

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

    def _train(self) -> None:
        """Local training loop with SGD and gradient clipping."""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.model.train()

        for _epoch in range(self.epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
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
