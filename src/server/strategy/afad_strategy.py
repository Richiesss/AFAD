import math
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

# FedGen regularization starts after this many warmup rounds.
# Models need ~6 rounds of independent training to reach 60-70% accuracy
# before generator-based regularization becomes beneficial.
FEDGEN_WARMUP_ROUNDS = 3


class AFADStrategy(fl.server.strategy.FedAvg):
    """
    AFAD Hybrid Strategy.

    Combines HeteroFL (intra-family, same architecture) and FedGen (inter-family)
    aggregation for heterogeneous federated learning.

    FedGen flow per round (server-side distillation):
    1. configure_fit: Send model params to clients
    2. Client fit: Standard local training (no FedGen regularization)
    3. aggregate_fit: HeteroFL aggregate → train generator → distill models
    4. configure_evaluate / aggregate_evaluate: Standard evaluation
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
        enable_fedgen: bool = True,
        enable_heterofl: bool = True,
        num_rounds: int = 20,
        num_classes: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.generator = initial_generator
        self.router = family_router
        self.model_factories = model_factories or {}
        self.enable_fedgen = enable_fedgen
        self.enable_heterofl = enable_heterofl
        self.num_rounds = num_rounds
        self.num_classes = num_classes

        # Per-signature global models (stored as numpy arrays)
        self.global_models: dict[str, list[np.ndarray]] = {}

        # Per-client models for FedGen-only mode (no HeteroFL aggregation)
        self.client_models: dict[str, list[np.ndarray]] = {}
        self.client_model_names: dict[str, str] = {}

        # Client model rates for HeteroFL
        self.client_model_rates: dict[str, float] = client_model_rates or {}

        # Client to family and signature mappings
        self.client_family_map: dict[str, str] = {}
        self.client_model_signatures: dict[str, str] = {}

        # Signature to model_name mapping (populated from client fit metrics)
        self.sig_to_model_name: dict[str, str] = {}

        # Per-client label counts for FedGen label weighting
        self.client_label_counts: dict[str, list[int]] = {}

        self.hetero_aggregator = HeteroFLAggregator()

        # FedGen configuration
        fedgen_cfg = fedgen_config or {}
        self.fedgen_distiller = FedGenDistiller(
            generator=self.generator,
            gen_lr=fedgen_cfg.get("gen_lr", 3e-4),
            batch_size=fedgen_cfg.get("batch_size", 128),
            ensemble_alpha=fedgen_cfg.get("ensemble_alpha", 1.0),
            ensemble_eta=fedgen_cfg.get("ensemble_eta", 1.0),
            device=fedgen_cfg.get("device", "cpu"),
            temperature=fedgen_cfg.get("temperature", 4.0),
            distill_lr=fedgen_cfg.get("distill_lr", 1e-4),
            distill_epochs=fedgen_cfg.get("distill_epochs", 1),
            distill_steps=fedgen_cfg.get("distill_steps", 5),
            distill_alpha=fedgen_cfg.get("distill_alpha", 1.0),
            distill_beta=fedgen_cfg.get("distill_beta", 0.1),
            distill_every=fedgen_cfg.get("distill_every", 2),
        )
        self.fedgen_gen_epochs = fedgen_cfg.get("gen_epochs", 1)
        self.fedgen_teacher_iters = fedgen_cfg.get("teacher_iters", 20)

        # Training config to propagate to clients
        self.training_config = training_config or {}

        # Metrics
        self.metrics_collector = MetricsCollector()

        # Track whether generator has been trained at least once
        self._generator_trained = False

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
        """Configure the next round of training, sending model params to clients."""
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

            # Only send family if actually known
            family = self.client_family_map.get(cid)
            if family:
                new_config["family"] = family

            # Propagate training config with cosine LR scheduling
            for key in ("momentum", "weight_decay", "local_epochs", "fedprox_mu"):
                if key in self.training_config:
                    new_config[key] = self.training_config[key]

            # Cosine annealing: lr decays from lr_max to lr_min over num_rounds
            base_lr = self.training_config.get("lr", 0.01)
            lr_min = base_lr * 0.01  # Decay to 1% of initial LR
            progress = server_round / self.num_rounds
            new_config["lr"] = lr_min + 0.5 * (base_lr - lr_min) * (
                1 + math.cos(math.pi * progress)
            )

            # Send matching model: per-client (FedGen-only) or per-signature
            sig = self.client_model_signatures.get(cid)
            if not self.enable_heterofl and cid in self.client_models:
                model_ndarrays = self.client_models[cid]
            elif sig and sig in self.global_models:
                model_ndarrays = self.global_models[sig]
            else:
                model_ndarrays = []
                new_config["use_local_init"] = True

            client_parameters = ndarrays_to_parameters(model_ndarrays)
            new_fit_ins.append((client, FitIns(client_parameters, new_config)))

        return new_fit_ins

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """
        Aggregate client updates with HeteroFL + FedGen generator training.

        1. Group clients by model signature
        2. Apply HeteroFL aggregation within each group
        3. Collect label counts from clients
        4. Train generator using aggregated models (FedGen server-side)
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

            # Parse and store label counts
            label_counts_str = fit_res.metrics.get("label_counts", "")
            if isinstance(label_counts_str, str) and label_counts_str:
                counts = [int(x) for x in label_counts_str.split(",") if x]
                self.client_label_counts[cid] = counts

            params = parameters_to_ndarrays(fit_res.parameters)

            sig = self._get_model_signature(params)
            self.client_model_signatures[cid] = sig

            # Track sig -> model_name and per-client model_name mappings
            if model_name:
                self.sig_to_model_name[sig] = model_name
                self.client_model_names[cid] = model_name

            if sig not in signature_results:
                signature_results[sig] = []
            signature_results[sig].append((cid, params, fit_res.num_examples, family))

        if self.enable_heterofl:
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
                    results=[
                        (params, num_ex) for _, params, num_ex in aggregation_items
                    ],
                    global_params=current_global,
                )
                self.global_models[sig] = parameters_to_ndarrays(updated_params)

                families = set(family for _, _, _, family in items)
                logger.info(
                    f"Signature {sig[:50]}...: aggregated {len(items)} clients "
                    f"from families {families}"
                )
        else:
            # FedGen-only: store per-client models, no intra-group averaging
            for sig, items in signature_results.items():
                for cid, params, _num_ex, _family in items:
                    self.client_models[cid] = [p.copy() for p in params]
                # Keep global_models for initial delivery only
                if sig not in self.global_models:
                    self.global_models[sig] = [p.copy() for p in items[0][1]]
            logger.info(f"FedGen-only mode: stored {len(results)} per-client models")

        # FedGen: Train generator + distill models after aggregation
        num_available = (
            len(self.client_models)
            if not self.enable_heterofl
            else len(self.global_models)
        )
        if self.enable_fedgen and num_available >= 2:
            self._train_generator_on_server(server_round)

        # Metrics
        metrics: dict[str, Scalar] = {
            "round": server_round,
            "num_signatures": len(signature_results),
            "total_clients": len(results),
        }

        # Return first model as "main" global for Flower
        if self.global_models:
            main_sig = list(self.global_models.keys())[0]
            return (
                ndarrays_to_parameters(self.global_models[main_sig]),
                metrics,
            )

        return None, metrics

    def _train_generator_on_server(self, server_round: int) -> None:
        """Reconstruct nn.Module models, train generator, and distill models."""
        if not self.model_factories:
            logger.warning("No model_factories provided; skipping generator training")
            return

        # Reconstruct nn.Module from stored numpy arrays
        torch_models: dict[str, nn.Module] = {}

        if not self.enable_heterofl and self.client_models:
            # FedGen-only: per-client models
            for cid, np_params in self.client_models.items():
                model_name = self.client_model_names.get(cid)
                if model_name and model_name in self.model_factories:
                    model = self.model_factories[model_name](
                        num_classes=self.num_classes
                    )
                    state_dict = model.state_dict()
                    keys = list(state_dict.keys())
                    if len(keys) == len(np_params):
                        for key, arr in zip(keys, np_params):
                            state_dict[key] = torch.from_numpy(arr.copy())
                        model.load_state_dict(state_dict)
                        model.to(self.fedgen_distiller.device)
                        torch_models[cid] = model
                    else:
                        logger.warning(
                            f"Param count mismatch for cid={cid}: "
                            f"model has {len(keys)}, stored {len(np_params)}"
                        )
                else:
                    logger.debug(f"No factory for cid={cid} (model_name={model_name})")
        else:
            # HeteroFL / AFAD: per-signature models
            for sig, np_params in self.global_models.items():
                model_name = self.sig_to_model_name.get(sig)
                if model_name and model_name in self.model_factories:
                    model = self.model_factories[model_name](
                        num_classes=self.num_classes
                    )
                    state_dict = model.state_dict()
                    keys = list(state_dict.keys())
                    if len(keys) == len(np_params):
                        for key, arr in zip(keys, np_params):
                            state_dict[key] = torch.from_numpy(arr.copy())
                        model.load_state_dict(state_dict)
                        model.to(self.fedgen_distiller.device)
                        torch_models[sig] = model
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
            logger.info("Not enough reconstructable models for generator training")
            return

        # Compute label weights
        if not self.enable_heterofl and self.client_models:
            # FedGen-only: per-client counts (naturally aligned with per-client models)
            label_counts_list = [
                self.client_label_counts[cid]
                for cid in torch_models
                if cid in self.client_label_counts
            ]
        else:
            # HeteroFL / AFAD: aggregate per-client counts to per-signature
            sig_label_counts: dict[str, list[int]] = {}
            for cid, counts in self.client_label_counts.items():
                sig = self.client_model_signatures.get(cid)
                if sig and sig in torch_models:
                    if sig not in sig_label_counts:
                        sig_label_counts[sig] = [0] * self.num_classes
                    for i, c in enumerate(counts):
                        if i < self.num_classes:
                            sig_label_counts[sig][i] += c
            label_counts_list = [
                sig_label_counts[sig] for sig in torch_models if sig in sig_label_counts
            ]

        if not label_counts_list:
            # Fallback: uniform weights
            num_models = len(torch_models)
            label_weights = np.ones((self.num_classes, num_models)) / num_models
            qualified_labels = list(range(self.num_classes))
        else:
            label_weights, qualified_labels = self.fedgen_distiller.get_label_weights(
                label_counts_list, num_classes=self.num_classes
            )

        # Train generator
        self.fedgen_distiller.train_generator(
            models=torch_models,
            label_weights=label_weights,
            qualified_labels=qualified_labels,
            num_epochs=self.fedgen_gen_epochs,
            num_teacher_iters=self.fedgen_teacher_iters,
        )

        self._generator_trained = True
        logger.info(f"Generator trained with {len(torch_models)} models")

        # Server-side knowledge distillation (after warmup, periodic)
        distill_every = self.fedgen_distiller.distill_every
        rounds_since_warmup = server_round - FEDGEN_WARMUP_ROUNDS
        if (
            server_round > FEDGEN_WARMUP_ROUNDS
            and rounds_since_warmup % distill_every == 0
        ):
            distill_losses = self.fedgen_distiller.distill_models(
                models=torch_models,
                label_weights=label_weights,
                qualified_labels=qualified_labels,
            )

            # Write distilled model params back
            if distill_losses:
                if not self.enable_heterofl:
                    # FedGen-only: write back to per-client models
                    for cid, model in torch_models.items():
                        self.client_models[cid] = [
                            val.cpu().detach().numpy()
                            for val in model.state_dict().values()
                        ]
                else:
                    # HeteroFL / AFAD: write back to global_models
                    for sig, model in torch_models.items():
                        self.global_models[sig] = [
                            val.cpu().detach().numpy()
                            for val in model.state_dict().values()
                        ]
                logger.info(
                    f"Distilled {len(distill_losses)} models, "
                    f"avg_loss={sum(distill_losses.values()) / len(distill_losses):.4f}"
                )

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
            if not self.enable_heterofl and cid in self.client_models:
                client_params = ndarrays_to_parameters(self.client_models[cid])
            elif sig and sig in self.global_models:
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
