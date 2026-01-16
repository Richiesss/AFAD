import flwr as fl
from typing import List, Tuple, Dict, Any, Union, Optional
from flwr.common import Parameters, Scalar, FitRes, FitIns, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from src.routing.family_router import FamilyRouter
from src.server.generator.synthetic_generator import SyntheticGenerator
from src.server.strategy.heterofl_aggregator import HeteroFLAggregator
from src.server.strategy.fedgen_distiller import FedGenDistiller
from src.models.registry import ModelRegistry
from src.utils.logger import setup_logger
import torch
import numpy as np

logger = setup_logger("AFADStrategy")

class AFADStrategy(fl.server.strategy.FedAvg):
    """
    AFAD Hybrid Strategy
    Combines HeteroFL (Intra-family) and FedGen (Inter-family).
    """
    def __init__(
        self,
        initial_parameters: Parameters,
        initial_generator: SyntheticGenerator,
        family_router: FamilyRouter,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.generator = initial_generator
        self.router = family_router
        
        # Family-specific global models (Stored as list of numpy arrays for Flower)
        # Assuming initial_parameters contains the CNN Global (since it matches standard FL).
        # We need a way to initialization multiple global models.
        # For now, we initialize CNN from 'initial_parameters' and ViT separately or lazily.
        # Actually, let's keep a dict of {family: ndarrays}.
        # In a real scenario, we'd initialize these properly.
        # For Phase 1, we start with CNN Global and ViT Global.
        
        self.global_models: Dict[str, List[np.ndarray]] = {}
        # We'll populate this on first round or assume provided.
        # To avoid complexity, we assume the first standard 'parameters' corresponds to CNN.
        # We can lazy-init ViT.
        
        self.hetero_aggregator = HeteroFLAggregator()
        self.fedgen_distiller = FedGenDistiller(self.generator) # Needs to be on device inside class

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager):
        # Triggered at start? Standard FedAvg returns initial_parameters.
        # We might need to override.
        return None # Let standard flow handle used kwargs 'initial_parameters'

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configure the next round of training.
        """
        logger.info(f"Round {server_round}: configure_fit")
        
        # Standard Feed: Sample clients
        config = {}
        standard_fit_ins = super().configure_fit(server_round, parameters, client_manager)
        if not standard_fit_ins:
            return []

        # Customize FitIns per client
        # In simulation, we might not know client ID easily in 'standard_fit_ins' list (it's ClientProxy).
        # We need to query client properties if possible, or assume config is passed.
        
        # Prepare Generator Weights to send
        gen_weights = [val.cpu().numpy() for val in self.generator.state_dict().values()]
        gen_params = ndarrays_to_parameters(gen_weights)
        
        # We need to split standard_fit_ins and assign correct family model + generator.
        # Since we don't know which client is which family here easily without a registry look-up 
        # (ClientProxy has cid, we can lookup if we have a table).
        # We'll assume the client manages its own architecture and we just send "Global Family Model" if we knew it.
        # BUT: The strategy *sends* the parameters.
        # If we send CNN params to ViT client, it breaks.
        # Solution: Maintain a map of CID -> Family.
        # In Phase 1, we can hardcode or learn from first 'get_properties'.
        
        # For simplicity in Phase 1:
        # We just pass the parameters we have. If it's the wrong family, the client updates *nothing* or crash?
        # AFAD requires Server to know Family.
        # Let's assume we have a `client_family_map` populated or config-based.
        # Since we don't have it yet, we just pass the 'Ensemble/Generator' info and let Client request model?
        # Flower Config:
        # We can put 'generator_params': gen_params in config (as bytes?).
        # Parameters object is not serializable in dict easily? It is Serializable.
        
        new_fit_ins = []
        for client, fit_ins in standard_fit_ins:
            # fit_ins.parameters contains 'parameters' passed to this function (which is global model).
            # If we maintain multiple models, 'parameters' arg is insufficient (it's only one).
            # We need to ignore it and inject correct one.
            
            cid = client.cid
            # Mock family lookup (should use registry or client info)
            # For now, let's assume we just use the global parameters as-is and hope for best (or single family start).
            # If multi-family, we must look up.
            
            # config update
            fit_ins.config['round'] = server_round
            # TODO: Add generator weights to config
            
            new_fit_ins.append((client, fit_ins))
            
        return new_fit_ins

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        logger.info(f"Round {server_round}: aggregate_fit")
        if not results:
            return None, {}
            
        # Group results by family
        family_results: Dict[str, List] = {}
        
        for client, fit_res in results:
            # Client returns metrics containing its family?
            family = fit_res.metrics.get("family", "unknown")
            if family not in family_results:
                family_results[family] = []
            
            # Convert parameters back to ndarrays
            params = parameters_to_ndarrays(fit_res.parameters)
            family_results[family].append((params, fit_res.num_examples))
            
        # Aggregate per family
        new_global_params_list = [] # Which one to return to Flower? 
        # Flower expects ONE set of parameters to be persisted as 'global'.
        # We can serialize our Dict[str, params] into one Parameters object (custom serialization)
        # or just return one of them (e.g. CNN) as 'main'.
        
        for family, items in family_results.items():
            current_global = self.global_models.get(family, [np.zeros_like(p) for p in items[0][0]]) # Init if empty
            
            updated_params = self.hetero_aggregator.aggregate(family, items, current_global)
            self.global_models[family] = parameters_to_ndarrays(updated_params)
            
        # Distill (FedGen)
        # Need to convert ndarrays back to torch models for distillation
        # self.fedgen_distiller.update(self.global_models)
        
        # Return Main model (e.g. CNN) as the "Global" one for Flower state
        # Or return empty if we manage state internally.
        
        # Metrics
        metrics = {"round": server_round}
        return None, metrics
