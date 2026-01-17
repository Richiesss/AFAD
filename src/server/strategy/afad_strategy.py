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
    Combines HeteroFL (Intra-family, same architecture) and FedGen (Inter-family).

    HeteroFL: Used when clients have the SAME architecture but different widths
    FedGen: Used for knowledge distillation across different architectures
    """
    def __init__(
        self,
        initial_parameters: Parameters,
        initial_generator: SyntheticGenerator,
        family_router: FamilyRouter,
        client_model_rates: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.generator = initial_generator
        self.router = family_router

        # Family-specific global models (Stored as list of numpy arrays for Flower)
        # Key: (family, model_signature) where model_signature is a tuple of layer shapes
        self.global_models: Dict[str, List[np.ndarray]] = {}

        # Client model rates for HeteroFL (client_id -> model_rate)
        self.client_model_rates: Dict[str, float] = client_model_rates or {}

        # Client to family mapping
        self.client_family_map: Dict[str, str] = {}

        # Client to model signature mapping (for grouping same-architecture clients)
        self.client_model_signatures: Dict[str, str] = {}

        self.hetero_aggregator = HeteroFLAggregator()
        self.fedgen_distiller = FedGenDistiller(self.generator)

    def set_client_model_rate(self, client_id: str, model_rate: float):
        """Set the model rate for a specific client"""
        self.client_model_rates[client_id] = model_rate

    def set_client_family(self, client_id: str, family: str):
        """Set the family for a specific client"""
        self.client_family_map[client_id] = family

    def _get_model_signature(self, params: List[np.ndarray]) -> str:
        """
        Create a signature from parameter shapes to identify same-architecture models.
        Models with the same signature have compatible parameter structures.
        """
        shapes = tuple(p.shape for p in params)
        return str(shapes)

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager):
        return None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configure the next round of training.

        For AFAD with heterogeneous architectures, we send each client
        its own model parameters (no HeteroFL distribution for different architectures).
        """
        logger.info(f"Round {server_round}: configure_fit")

        # Standard Feed: Sample clients
        standard_fit_ins = super().configure_fit(server_round, parameters, client_manager)
        if not standard_fit_ins:
            return []

        new_fit_ins = []
        for client, fit_ins in standard_fit_ins:
            cid = client.cid
            family = self.client_family_map.get(cid, "default")

            new_config = dict(fit_ins.config)
            new_config['round'] = server_round
            new_config['family'] = family

            # Check if we have stored parameters for this client's model signature
            sig = self.client_model_signatures.get(cid)
            if sig and sig in self.global_models:
                # Send the matching global model
                client_parameters = ndarrays_to_parameters(self.global_models[sig])
            else:
                # First round or new client: send empty parameters
                # Client will use its own initialized model
                client_parameters = ndarrays_to_parameters([])
                new_config['use_local_init'] = True

            new_fit_ins.append((client, FitIns(client_parameters, new_config)))

        return new_fit_ins

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate client updates.

        Strategy:
        1. Group clients by model signature (same architecture)
        2. Apply HeteroFL aggregation within each signature group
        3. Different architectures are kept separate (for future FedGen distillation)
        """
        logger.info(f"Round {server_round}: aggregate_fit")
        if not results:
            return None, {}

        # Group results by model signature (same architecture)
        signature_results: Dict[str, List[Tuple[str, List[np.ndarray], int, str]]] = {}

        for client, fit_res in results:
            cid = client.cid
            family = fit_res.metrics.get("family", self.client_family_map.get(cid, "default"))
            params = parameters_to_ndarrays(fit_res.parameters)

            # Get model signature
            sig = self._get_model_signature(params)
            self.client_model_signatures[cid] = sig

            if sig not in signature_results:
                signature_results[sig] = []

            signature_results[sig].append((cid, params, fit_res.num_examples, family))

        # Aggregate per model signature using HeteroFL
        for sig, items in signature_results.items():
            # Get current global model for this signature
            if sig not in self.global_models:
                # Initialize from first client's params
                first_params = items[0][1]
                self.global_models[sig] = [p.copy() for p in first_params]

            current_global = self.global_models[sig]

            # Convert items to format expected by aggregate
            aggregation_items = [(cid, params, num_ex) for cid, params, num_ex, _ in items]

            # For same-architecture models, use HeteroFL aggregation
            # (currently all clients have model_rate=1.0, so this is like FedAvg)
            updated_params = self.hetero_aggregator.aggregate_simple(
                family=sig,
                results=[(params, num_ex) for _, params, num_ex in aggregation_items],
                global_params=current_global
            )
            self.global_models[sig] = parameters_to_ndarrays(updated_params)

            # Log aggregation info
            families = set(family for _, _, _, family in items)
            logger.info(f"Signature {sig[:50]}...: aggregated {len(items)} clients from families {families}")

        # Metrics
        metrics = {
            "round": server_round,
            "num_signatures": len(signature_results),
            "total_clients": len(results)
        }

        # Return the first model as the "main" global for Flower
        if self.global_models:
            main_sig = list(self.global_models.keys())[0]
            return ndarrays_to_parameters(self.global_models[main_sig]), metrics

        return None, metrics
