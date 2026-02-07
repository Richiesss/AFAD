from collections.abc import Callable
from typing import Any

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from src.routing.family_router import FamilyRouter
from src.server.generator.synthetic_generator import SyntheticGenerator
from src.server.strategy.fedgen_distiller import FedGenDistiller
from src.server.strategy.heterofl_aggregator import HeteroFLAggregator
from src.utils.logger import setup_logger
from src.utils.metrics import MetricsCollector

logger = setup_logger("AFADStrategy")


class AFADStrategy(fl.server.strategy.FedAvg):
    """
    AFAD Hybrid Strategy.

    Combines HeteroFL (intra-family, same architecture) and FedGen (inter-family)
    aggregation for heterogeneous federated learning.
    """

    def __init__(
        self,
        initial_parameters: Parameters,
        initial_generator: SyntheticGenerator,
        family_router: FamilyRouter,
        model_factories: dict[str, Callable[..., nn.Module]] | None = None,
        client_model_rates: dict[str, float] | None = None,
        fedgen_config: dict[str, Any] | None = None,
        training_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.generator = initial_generator
        self.router = family_router
        self.model_factories = model_factories or {}

        # Per-signature global models (stored as numpy arrays)
        self.global_models: dict[str, list[np.ndarray]] = {}

        # Client model rates for HeteroFL
        self.client_model_rates: dict[str, float] = client_model_rates or {}

        # Client to family and signature mappings
        self.client_family_map: dict[str, str] = {}
        self.client_model_signatures: dict[str, str] = {}

        # Signature to model_name mapping (populated from client fit metrics)
        self.sig_to_model_name: dict[str, str] = {}

        self.hetero_aggregator = HeteroFLAggregator()

        # FedGen configuration
        fedgen_cfg = fedgen_config or {}
        self.fedgen_distiller = FedGenDistiller(
            generator=self.generator,
            temperature=fedgen_cfg.get("temperature", 4.0),
            gen_lr=fedgen_cfg.get("gen_lr", 0.001),
            distill_lr=fedgen_cfg.get("distill_lr", 0.001),
            batch_size=fedgen_cfg.get("batch_size", 32),
            device=fedgen_cfg.get("device", "cpu"),
        )
        self.fedgen_gen_steps = fedgen_cfg.get("gen_steps", 10)
        self.fedgen_distill_steps = fedgen_cfg.get("distill_steps", 5)

        # Training config to propagate to clients
        self.training_config = training_config or {}

        # Metrics
        self.metrics_collector = MetricsCollector()

    def set_client_model_rate(self, client_id: str, model_rate: float) -> None:
        self.client_model_rates[client_id] = model_rate

    def set_client_family(self, client_id: str, family: str) -> None:
        self.client_family_map[client_id] = family

    def _get_model_signature(self, params: list[np.ndarray]) -> str:
        """Create a signature from parameter shapes to identify same-architecture models."""
        shapes = tuple(p.shape for p in params)
        return str(shapes)

    def initialize_parameters(
        self, client_manager: fl.server.client_manager.ClientManager
    ):
        return None

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        logger.info(f"Round {server_round}: configure_fit")
        self.metrics_collector.start_round()

        standard_fit_ins = super().configure_fit(
            server_round, parameters, client_manager
        )
        if not standard_fit_ins:
            return []

        new_fit_ins = []
        for client, fit_ins in standard_fit_ins:
            cid = client.cid

            new_config = dict(fit_ins.config)
            new_config["round"] = server_round

            # Only send family if actually known (avoid overriding client's own family)
            family = self.client_family_map.get(cid)
            if family:
                new_config["family"] = family

            # Propagate training config
            for key in ("lr", "momentum", "weight_decay", "local_epochs"):
                if key in self.training_config:
                    new_config[key] = self.training_config[key]

            # Send matching global model if available
            sig = self.client_model_signatures.get(cid)
            if sig and sig in self.global_models:
                client_parameters = ndarrays_to_parameters(self.global_models[sig])
            else:
                client_parameters = ndarrays_to_parameters([])
                new_config["use_local_init"] = True

            new_fit_ins.append((client, FitIns(client_parameters, new_config)))

        return new_fit_ins

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """
        Aggregate client updates with HeteroFL + FedGen distillation.

        1. Group clients by model signature
        2. Apply HeteroFL aggregation within each group
        3. Run FedGen inter-family distillation (after round 1)
        """
        logger.info(f"Round {server_round}: aggregate_fit")
        if not results:
            return None, {}

        # Group by model signature
        signature_results: dict[str, list[tuple[str, list[np.ndarray], int, str]]] = {}

        for client, fit_res in results:
            cid = client.cid
            family = fit_res.metrics.get(
                "family", self.client_family_map.get(cid, "default")
            )
            model_name = fit_res.metrics.get("model_name", "")

            # Track client family for future rounds
            if family and family != "default":
                self.client_family_map[cid] = family
            params = parameters_to_ndarrays(fit_res.parameters)

            sig = self._get_model_signature(params)
            self.client_model_signatures[cid] = sig

            # Track sig -> model_name mapping
            if model_name:
                self.sig_to_model_name[sig] = model_name

            if sig not in signature_results:
                signature_results[sig] = []
            signature_results[sig].append((cid, params, fit_res.num_examples, family))

        # HeteroFL aggregation per signature group
        for sig, items in signature_results.items():
            if sig not in self.global_models:
                first_params = items[0][1]
                self.global_models[sig] = [p.copy() for p in first_params]

            current_global = self.global_models[sig]
            aggregation_items = [
                (cid, params, num_ex) for cid, params, num_ex, _ in items
            ]

            updated_params = self.hetero_aggregator.aggregate_simple(
                family=sig,
                results=[(params, num_ex) for _, params, num_ex in aggregation_items],
                global_params=current_global,
            )
            self.global_models[sig] = parameters_to_ndarrays(updated_params)

            families = set(family for _, _, _, family in items)
            logger.info(
                f"Signature {sig[:50]}...: aggregated {len(items)} clients "
                f"from families {families}"
            )

        # FedGen distillation (after warmup rounds, when >= 2 unique models exist)
        FEDGEN_WARMUP_ROUNDS = 3
        if server_round > FEDGEN_WARMUP_ROUNDS and len(self.global_models) >= 2:
            self._run_fedgen_distillation()

        # Metrics
        metrics: dict[str, Scalar] = {
            "round": server_round,
            "num_signatures": len(signature_results),
            "total_clients": len(results),
        }

        # Return first model as "main" global for Flower
        if self.global_models:
            main_sig = list(self.global_models.keys())[0]
            return ndarrays_to_parameters(self.global_models[main_sig]), metrics

        return None, metrics

    def _run_fedgen_distillation(self) -> None:
        """Reconstruct nn.Module models from numpy arrays, run FedGen, write back."""
        if not self.model_factories:
            logger.warning("No model_factories provided; skipping FedGen distillation")
            return

        # Reconstruct nn.Module from stored numpy arrays
        torch_models: dict[str, nn.Module] = {}
        sig_order: list[str] = []

        for sig, np_params in self.global_models.items():
            model_name = self.sig_to_model_name.get(sig)
            if model_name and model_name in self.model_factories:
                model = self.model_factories[model_name](num_classes=10)
                # Load numpy params into model
                state_dict = model.state_dict()
                keys = list(state_dict.keys())
                if len(keys) == len(np_params):
                    for key, arr in zip(keys, np_params):
                        state_dict[key] = torch.from_numpy(arr.copy())
                    model.load_state_dict(state_dict)
                    torch_models[sig] = model
                    sig_order.append(sig)
                else:
                    logger.warning(
                        f"Param count mismatch for sig={sig[:30]}...: "
                        f"model has {len(keys)}, stored {len(np_params)}"
                    )
            else:
                logger.debug(
                    f"No factory for sig={sig[:30]}... (model_name={model_name})"
                )

        if len(torch_models) < 2:
            logger.info("Not enough reconstructable models for FedGen distillation")
            return

        # Run FedGen distillation
        updated_models = self.fedgen_distiller.distill(
            torch_models,
            gen_steps=self.fedgen_gen_steps,
            distill_steps=self.fedgen_distill_steps,
        )

        # Write back updated parameters
        for sig, model in updated_models.items():
            updated_np = [
                val.cpu().detach().numpy() for val in model.state_dict().values()
            ]
            self.global_models[sig] = updated_np

        logger.info(f"FedGen distillation updated {len(updated_models)} models")

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Send each client its own model for evaluation."""
        standard_eval_ins = super().configure_evaluate(
            server_round, parameters, client_manager
        )
        if not standard_eval_ins:
            return []

        new_eval_ins = []
        for client, eval_ins in standard_eval_ins:
            cid = client.cid
            config = dict(eval_ins.config)
            config["round"] = server_round

            sig = self.client_model_signatures.get(cid)
            if sig and sig in self.global_models:
                client_params = ndarrays_to_parameters(self.global_models[sig])
            else:
                client_params = eval_ins.parameters

            new_eval_ins.append((client, EvaluateIns(client_params, config)))

        return new_eval_ins

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        """Aggregate per-client evaluation metrics."""
        if not results:
            return None, {}

        total_loss = 0.0
        total_accuracy = 0.0
        total_examples = 0

        for _client, eval_res in results:
            num_examples = eval_res.num_examples
            total_loss += eval_res.loss * num_examples
            total_accuracy += eval_res.metrics.get("accuracy", 0.0) * num_examples
            total_examples += num_examples

        avg_loss = total_loss / total_examples if total_examples > 0 else 0.0
        avg_accuracy = total_accuracy / total_examples if total_examples > 0 else 0.0

        # Record metrics
        self.metrics_collector.record_round(
            round_num=server_round,
            loss=avg_loss,
            accuracy=avg_accuracy,
            num_clients=len(results),
        )

        metrics: dict[str, Scalar] = {
            "round": server_round,
            "accuracy": avg_accuracy,
            "loss": avg_loss,
            "num_clients": len(results),
        }

        logger.info(
            f"Round {server_round} evaluation: "
            f"accuracy={avg_accuracy:.4f}, loss={avg_loss:.4f}"
        )

        return avg_loss, metrics
