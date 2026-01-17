import flwr as fl
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from src.utils.logger import setup_logger
from collections import OrderedDict
import numpy as np

logger = setup_logger("AFADClient")


class AFADClient(fl.client.NumPyClient):
    """
    AFAD Client Implementation with HeteroFL support

    Supports receiving sub-models based on model_rate and
    properly applying parameters for heterogeneous model widths.
    """
    def __init__(
        self,
        cid: str,
        model: nn.Module,
        train_loader,
        epochs: int,
        device: str = "cpu",
        family: str = "default",
        model_rate: float = 1.0
    ):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.epochs = epochs
        self.device = torch.device(device)
        self.model.to(self.device)
        self.family = family
        self.model_rate = model_rate  # Client's model rate for HeteroFL

        logger.info(f"Client {cid} initialized on device: {self.device}, "
                   f"family: {family}, model_rate: {model_rate}")

    def get_parameters(self, config) -> List[np.ndarray]:
        """Return model parameters as a list of NumPy arrays"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray], config: Dict = None):
        """
        Set model parameters from a list of NumPy arrays.

        Handles HeteroFL sub-models where parameters may have different shapes
        than the local model (when model_rate < 1.0).
        """
        model_keys = list(self.model.state_dict().keys())

        if len(parameters) != len(model_keys):
            logger.warning(f"Client {self.cid}: Parameter count mismatch. "
                          f"Received {len(parameters)}, model has {len(model_keys)}")

        state_dict = OrderedDict()
        for i, (key, param) in enumerate(zip(model_keys, parameters)):
            local_shape = self.model.state_dict()[key].shape
            received_shape = param.shape

            if local_shape == received_shape:
                # Shapes match - direct assignment
                state_dict[key] = torch.tensor(param)
            else:
                # HeteroFL: received sub-model parameters
                # Sub-model params are smaller and should be placed at the beginning
                logger.debug(f"Client {self.cid}: Layer {key} shape mismatch. "
                           f"Local: {local_shape}, Received: {received_shape}")

                # Start with zeros (or could use current weights)
                new_param = torch.zeros(local_shape)

                # Copy received params to the beginning (top-left for multi-dim)
                if len(local_shape) == 1:
                    new_param[:received_shape[0]] = torch.tensor(param)
                elif len(local_shape) == 2:
                    new_param[:received_shape[0], :received_shape[1]] = torch.tensor(param)
                elif len(local_shape) == 4:
                    new_param[:received_shape[0], :received_shape[1], :, :] = torch.tensor(param)
                else:
                    # Fallback: try to fit what we can
                    slices = tuple(slice(0, min(l, r)) for l, r in zip(local_shape, received_shape))
                    new_param[slices] = torch.tensor(param[slices])

                state_dict[key] = new_param

        self.model.load_state_dict(state_dict, strict=False)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train the model on local data.

        Args:
            parameters: Model parameters from server (may be sub-model or empty)
            config: Configuration dict containing round, model_rate, family, etc.

        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        # Update model_rate and family from config if provided
        if "model_rate" in config:
            self.model_rate = config["model_rate"]
        if "family" in config:
            self.family = config["family"]

        # Set parameters if provided and not using local initialization
        use_local_init = config.get("use_local_init", False)
        if parameters and len(parameters) > 0 and not use_local_init:
            self.set_parameters(parameters, config)
        # else: use locally initialized model (first round or heterogeneous setup)

        # Update local epochs from config
        if "local_epochs" in config:
            self.epochs = config["local_epochs"]

        # Train
        self._train()

        # Return updated weights
        return (
            self.get_parameters(config={}),
            len(self.train_loader.dataset),
            {
                "family": self.family,
                "model_rate": self.model_rate,
                "client_id": self.cid
            }
        )

    def _train(self):
        """
        Local training loop
        """
        from tqdm import tqdm

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.model.train()

        for epoch in range(self.epochs):
            with tqdm(self.train_loader, unit="batch",
                     desc=f"Client {self.cid} Epoch {epoch+1}/{self.epochs}",
                     leave=False) as tepoch:
                for images, labels in tepoch:
                    images, labels = images.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    tepoch.set_postfix(loss=loss.item())

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """
        Evaluate the model on local data.

        Args:
            parameters: Model parameters from server
            config: Configuration dict

        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        if parameters:
            self.set_parameters(parameters, config)

        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.train_loader:
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
                "model_rate": self.model_rate
            }
        )
