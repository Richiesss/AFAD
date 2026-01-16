import flwr as fl
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from src.utils.logger import setup_logger
from collections import OrderedDict

logger = setup_logger("AFADClient")

class AFADClient(fl.client.NumPyClient):
    """
    AFAD Client Implementation
    """
    def __init__(self, cid: str, model: nn.Module, train_loader, epochs: int, device: str = "cpu"):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.epochs = epochs
        self.device = torch.device(device)
        self.model.to(self.device)
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        logger.info(f"Client {self.cid}: fit()")
        
        # Set parameters (Simple version for now)
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        
        # Local training loop (Placeholder)
        # self._train()
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        logger.info(f"Client {self.cid}: evaluate()")
        # Evaluation logic (Placeholder)
        return 0.0, len(self.train_loader.dataset), {"accuracy": 0.0}

    def _train(self):
        """
        Local training loop
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.model.train()
        
        for epoch in range(self.epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
    def fit(self, parameters, config):
        # logger.info(f"Client {self.cid}: fit()")
        
        # Update local model weights with global weights
        # Note: parameters here comes from Strategy. If we use HeteroFL, 
        # parameters might be partial or full.
        # For simplicity in Phase 1, assuming Full parameters or compatible shape.
        if parameters:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            try:
                self.model.load_state_dict(state_dict, strict=False) # Allow partial loading for heterogeneous setup
            except RuntimeError as e:
                logger.warning(f"Client {self.cid}: Parameter mismatch during load (expected for heterogeneous setups): {e}")
            
        # Update hyperparameters
        if "local_epochs" in config:
            self.epochs = config["local_epochs"]
            
        # Train
        self._train()
        
        # Return updated weights
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"family": "unknown"} # TODO pass family

