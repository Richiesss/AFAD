"""AFAD Strategy: HeteroFL + FedGen hybrid federated learning.

Three operating modes:
- HeteroFL Only: Width-scaled sub-models, intra-family aggregation, no KD
- FedGen Only: Same-size models, FedAvg, client-side latent KD
- AFAD Hybrid: Width-scaled sub-models + intra-family aggregation + latent KD

Per-round flow (AFAD Hybrid):
1. configure_fit: HeteroFL distribute sub-models + send generator params
2. Client fit: Local training with FedGen regularization
3. aggregate_fit: HeteroFL aggregate per family + train generator
4. configure_evaluate / aggregate_evaluate: Standard evaluation
"""

import math
import pickle
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

from src.server.generator.afad_generator_trainer import AFADGeneratorTrainer
from src.server.generator.fedgen_generator import FedGenGenerator
from src.server.strategy.heterofl_aggregator import HeteroFLAggregator
from src.utils.logger import setup_logger
from src.utils.metrics import MetricsCollector

logger = setup_logger("AFADStrategy")

# FedGen regularization starts after this many warmup rounds.
FEDGEN_WARMUP_ROUNDS = 3


class AFADStrategy(fl.server.strategy.FedAvg):
    """AFAD Hybrid Strategy.

    Combines HeteroFL (intra-family width-scaling) and FedGen
    (inter-family latent-space KD) for heterogeneous FL.

    Args:
        initial_parameters: Initial global model parameters.
        generator: FedGenGenerator for latent-space KD (None = no KD).
        model_factories: Dict mapping model_name to factory function.
            For AFAD/FedGen modes, factories must create FedGenModelWrapper
            instances. For HeteroFL Only, plain models.
        client_model_rates: Dict mapping client_id to HeteroFL width rate.
        family_model_names: Dict mapping family to model_name in registry.
        fedgen_config: Generator training hyperparameters.
        training_config: Client training hyperparameters.
        enable_fedgen: Enable FedGen latent-space KD.
        enable_heterofl: Enable HeteroFL width-scaling aggregation.
        num_rounds: Total number of FL rounds.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        initial_parameters: Parameters,
        generator: FedGenGenerator | None = None,
        model_factories: dict[str, Callable[..., nn.Module]] | None = None,
        client_model_rates: dict[str, float] | None = None,
        family_model_names: dict[str, str] | None = None,
        fedgen_config: dict[str, Any] | None = None,
        training_config: dict[str, Any] | None = None,
        enable_fedgen: bool = True,
        enable_heterofl: bool = True,
        num_rounds: int = 20,
        num_classes: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.generator = generator
        self.model_factories = model_factories or {}
        self.enable_fedgen = enable_fedgen
        self.enable_heterofl = enable_heterofl
        self.num_rounds = num_rounds
        self.num_classes = num_classes

        # Per-family global models (rate=1.0 params as numpy arrays)
        self.family_global_models: dict[str, list[np.ndarray]] = {}
        self.family_model_names: dict[str, str] = family_model_names or {}

        # Client model rates for HeteroFL width-scaling
        self.client_model_rates: dict[str, float] = client_model_rates or {}

        # Client to family mapping
        self.client_family_map: dict[str, str] = {}

        # Per-client label counts for FedGen label weighting
        self.client_label_counts: dict[str, list[int]] = {}

        # HeteroFL aggregator
        self.hetero_aggregator = HeteroFLAggregator()

        # Number of tail layers to preserve during HeteroFL distribute/aggregate.
        # 1 = classifier only (vanilla HeteroFL)
        # 2 = bottleneck + classifier (AFAD with FedGenModelWrapper)
        self._num_preserved_tail_layers = 2 if enable_fedgen else 1

        # FedGen generator trainer (server-side, latent-space)
        fedgen_cfg = fedgen_config or {}
        self._device = fedgen_cfg.get("device", "cpu")
        if self.generator is not None and self.enable_fedgen:
            self.generator_trainer = AFADGeneratorTrainer(
                generator=self.generator,
                gen_lr=fedgen_cfg.get("gen_lr", 3e-4),
                batch_size=fedgen_cfg.get("batch_size", 128),
                ensemble_alpha=fedgen_cfg.get("ensemble_alpha", 1.0),
                ensemble_eta=fedgen_cfg.get("ensemble_eta", 1.0),
                device=self._device,
            )
        else:
            self.generator_trainer = None
        self._gen_epochs = fedgen_cfg.get("gen_epochs", 1)
        self._gen_teacher_iters = fedgen_cfg.get("teacher_iters", 20)

        # Training config to propagate to clients
        self.training_config = training_config or {}

        # Metrics
        self.metrics_collector = MetricsCollector()

        # Track whether generator has been trained at least once
        self._generator_trained = False

        # For Flower's return value (keep one set of params)
        self._global_params_for_flower: list[np.ndarray] | None = None

    # ─── Public helpers ───────────────────────────────────────────

    def set_client_model_rate(self, client_id: str, model_rate: float) -> None:
        self.client_model_rates[client_id] = model_rate

    def set_client_family(self, client_id: str, family: str) -> None:
        self.client_family_map[client_id] = family

    # ─── Internal helpers ─────────────────────────────────────────

    def _get_label_split(self, cid: str) -> list[int] | None:
        """Get non-zero label indices for a client (label-split aggregation)."""
        counts = self.client_label_counts.get(cid)
        if counts is None:
            return None
        return [i for i, c in enumerate(counts) if c > 0]

    def _initialize_family_models(self) -> None:
        """Create rate=1.0 global models for each family (called once)."""
        if self.family_global_models:
            return

        families_seen: set[str] = set()
        for _cid, family in self.client_family_map.items():
            if family in families_seen:
                continue
            families_seen.add(family)

            model_name = self.family_model_names.get(family)
            if model_name and model_name in self.model_factories:
                model = self.model_factories[model_name](num_classes=self.num_classes)
                params = [val.cpu().numpy() for val in model.state_dict().values()]
                self.family_global_models[family] = params
                logger.info(
                    f"Initialized family global model: {family} -> {model_name} "
                    f"({len(params)} params)"
                )

    def _reconstruct_model(
        self, model_name: str, np_params: list[np.ndarray]
    ) -> nn.Module | None:
        """Reconstruct nn.Module from stored numpy arrays."""
        if model_name not in self.model_factories:
            return None

        model = self.model_factories[model_name](num_classes=self.num_classes)
        state_dict = model.state_dict()
        keys = list(state_dict.keys())

        if len(keys) != len(np_params):
            logger.warning(
                f"Param count mismatch for {model_name}: "
                f"model has {len(keys)}, stored {len(np_params)}"
            )
            return None

        for key, arr in zip(keys, np_params):
            state_dict[key] = torch.from_numpy(arr.copy())
        model.load_state_dict(state_dict)
        model.to(self._device)
        return model

    def _serialize_generator_params(self) -> bytes | None:
        """Serialize generator params as bytes for Flower config."""
        if self.generator is None:
            return None
        gen_params = [val.cpu().numpy() for val in self.generator.state_dict().values()]
        return pickle.dumps(gen_params)

    # ─── Flower overrides ─────────────────────────────────────────

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
        """Configure training: distribute sub-models + generator params."""
        logger.info(f"Round {server_round}: configure_fit")
        self.metrics_collector.start_round()

        # Initialize family global models on first round
        if self.enable_heterofl:
            self._initialize_family_models()

        standard_fit_ins = super().configure_fit(
            server_round, parameters, client_manager
        )
        if not standard_fit_ins:
            return []

        # Serialize generator params once for all clients
        gen_params_bytes = None
        if (
            self.enable_fedgen
            and self._generator_trained
            and server_round > FEDGEN_WARMUP_ROUNDS
        ):
            gen_params_bytes = self._serialize_generator_params()

        new_fit_ins = []
        for client, fit_ins in standard_fit_ins:
            cid = client.cid

            new_config: dict[str, Scalar] = dict(fit_ins.config)
            new_config["round"] = server_round

            family = self.client_family_map.get(cid)
            if family:
                new_config["family"] = family

            # Propagate training config with cosine LR scheduling
            for key in ("momentum", "weight_decay", "local_epochs", "fedprox_mu"):
                if key in self.training_config:
                    new_config[key] = self.training_config[key]

            base_lr = self.training_config.get("lr", 0.01)
            lr_min = base_lr * 0.01
            progress = server_round / self.num_rounds
            new_config["lr"] = lr_min + 0.5 * (base_lr - lr_min) * (
                1 + math.cos(math.pi * progress)
            )

            # Enable/disable FedGen regularization on clients
            new_config["regularization"] = (
                self.enable_fedgen
                and self._generator_trained
                and server_round > FEDGEN_WARMUP_ROUNDS
            )

            # Send generator params to clients (AFAD/FedGen modes)
            if gen_params_bytes is not None:
                new_config["generator_params"] = gen_params_bytes

            # Determine model parameters to send
            if self.enable_heterofl and family and family in self.family_global_models:
                # HeteroFL mode: distribute sub-model from family global
                model_rate = self.client_model_rates.get(cid, 1.0)
                new_config["model_rate"] = model_rate
                family_global = self.family_global_models[family]
                model_ndarrays = self.hetero_aggregator.distribute(
                    family_global,
                    cid,
                    model_rate,
                    num_preserved_tail_layers=self._num_preserved_tail_layers,
                )
            else:
                # FedGen Only or first round: use family global (rate=1.0)
                if family and family in self.family_global_models:
                    model_ndarrays = self.family_global_models[family]
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
        """Aggregate client updates: HeteroFL per family + generator training."""
        logger.info(f"Round {server_round}: aggregate_fit")
        if not results:
            return None, {}

        # Group results by family
        family_results: dict[str, list[tuple[str, list[np.ndarray], int]]] = {}

        for client, fit_res in results:
            cid = client.cid
            family = fit_res.metrics.get(
                "family", self.client_family_map.get(cid, "default")
            )

            # Track client family
            if family and family != "default":
                self.client_family_map[cid] = family

            # Parse and store label counts
            label_counts_str = fit_res.metrics.get("label_counts", "")
            if isinstance(label_counts_str, str) and label_counts_str:
                counts = [int(x) for x in label_counts_str.split(",") if x]
                self.client_label_counts[cid] = counts

            params = parameters_to_ndarrays(fit_res.parameters)

            if family not in family_results:
                family_results[family] = []
            family_results[family].append((cid, params, fit_res.num_examples))

        # Aggregate per family
        if self.enable_heterofl:
            self._aggregate_heterofl(family_results)
        else:
            self._aggregate_fedavg(family_results)

        # Train generator after aggregation (AFAD/FedGen modes)
        if (
            self.enable_fedgen
            and self.generator_trainer is not None
            and len(self.family_global_models) >= 2
        ):
            self._train_generator_on_server(server_round)

        # Metrics
        metrics: dict[str, Scalar] = {
            "round": server_round,
            "num_families": len(family_results),
            "total_clients": len(results),
        }

        # Return first family model for Flower framework
        if self.family_global_models:
            first_family = next(iter(self.family_global_models))
            return (
                ndarrays_to_parameters(self.family_global_models[first_family]),
                metrics,
            )

        return None, metrics

    def _aggregate_heterofl(
        self,
        family_results: dict[str, list[tuple[str, list[np.ndarray], int]]],
    ) -> None:
        """HeteroFL aggregation: count-based within each family."""
        for family, items in family_results.items():
            if family not in self.family_global_models:
                logger.warning(f"Family {family} not in family_global_models, skipping")
                continue

            family_global = self.family_global_models[family]

            # Build label splits for output layer aggregation
            client_label_splits: dict[str, list[int]] = {}
            for cid, _params, _num_ex in items:
                label_split = self._get_label_split(cid)
                if label_split is not None:
                    client_label_splits[cid] = label_split

            updated_params = self.hetero_aggregator.aggregate(
                family=family,
                results=items,
                global_params=family_global,
                client_label_splits=client_label_splits or None,
                num_preserved_tail_layers=self._num_preserved_tail_layers,
            )
            self.family_global_models[family] = parameters_to_ndarrays(updated_params)

            logger.info(f"Family {family}: HeteroFL aggregated {len(items)} clients")

    def _aggregate_fedavg(
        self,
        family_results: dict[str, list[tuple[str, list[np.ndarray], int]]],
    ) -> None:
        """FedAvg aggregation: simple weighted average within each family."""
        for family, items in family_results.items():
            total_examples = sum(n for _, _, n in items)
            if total_examples == 0:
                continue

            # Initialize with zeros (float64 for weighted averaging)
            first_params = items[0][1]
            avg_params = [np.zeros_like(p, dtype=np.float64) for p in first_params]

            for _cid, params, num_ex in items:
                weight = num_ex / total_examples
                for i, p in enumerate(params):
                    avg_params[i] += weight * p

            self.family_global_models[family] = avg_params

            logger.info(f"Family {family}: FedAvg aggregated {len(items)} clients")

    def _train_generator_on_server(self, server_round: int) -> None:
        """Reconstruct FedGenModelWrapper models and train generator."""
        if not self.model_factories:
            logger.warning("No model_factories; skipping generator training")
            return

        # Reconstruct full-rate FedGenModelWrapper from family_global_models
        torch_models: dict[str, nn.Module] = {}
        for family, np_params in self.family_global_models.items():
            model_name = self.family_model_names.get(family)
            if not model_name:
                continue
            model = self._reconstruct_model(model_name, np_params)
            if model is not None:
                torch_models[family] = model

        if len(torch_models) < 2:
            logger.info("Not enough models for generator training")
            return

        # Compute label weights (aggregate per-client counts to per-family)
        family_label_counts: dict[str, list[int]] = {}
        for cid, counts in self.client_label_counts.items():
            family = self.client_family_map.get(cid)
            if family and family in torch_models:
                if family not in family_label_counts:
                    family_label_counts[family] = [0] * self.num_classes
                for i, c in enumerate(counts):
                    if i < self.num_classes:
                        family_label_counts[family][i] += c

        label_counts_list = [
            family_label_counts[f] for f in torch_models if f in family_label_counts
        ]

        if not label_counts_list:
            num_models = len(torch_models)
            label_weights = np.ones((self.num_classes, num_models)) / num_models
            qualified_labels = list(range(self.num_classes))
        else:
            label_weights, qualified_labels = self.generator_trainer.get_label_weights(
                label_counts_list, num_classes=self.num_classes
            )

        # Train generator using forward_from_latent
        self.generator_trainer.train_generator(
            models=torch_models,
            label_weights=label_weights,
            qualified_labels=qualified_labels,
            num_epochs=self._gen_epochs,
            num_teacher_iters=self._gen_teacher_iters,
        )

        self._generator_trained = True
        logger.info(f"Generator trained with {len(torch_models)} family models")

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

            family = self.client_family_map.get(cid)
            if self.enable_heterofl and family and family in self.family_global_models:
                model_rate = self.client_model_rates.get(cid, 1.0)
                family_global = self.family_global_models[family]
                sub_params = self.hetero_aggregator.distribute(
                    family_global,
                    cid,
                    model_rate,
                    num_preserved_tail_layers=self._num_preserved_tail_layers,
                )
                client_params = ndarrays_to_parameters(sub_params)
            elif family and family in self.family_global_models:
                # FedGen Only: send full-rate model
                client_params = ndarrays_to_parameters(
                    self.family_global_models[family]
                )
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
