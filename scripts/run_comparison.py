"""
Compare HeteroFL-only, FedGen-only, and AFAD hybrid (HeteroFL + FedGen).

Runs 3 Flower simulations sequentially with the same data split and
collects per-round accuracy/loss for each method.

Usage:
    python scripts/run_comparison.py                          # Phase 1 (MNIST)
    python scripts/run_comparison.py config/afad_phase2_config.yaml  # Phase 2
"""

import os
import sys

import flwr as fl
import numpy as np
import ray
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.models.cnn.heterofl_resnet  # noqa: F401 - register models
import src.models.cnn.mobilenet  # noqa: F401
import src.models.cnn.resnet  # noqa: F401
import src.models.vit.deit  # noqa: F401
import src.models.vit.heterofl_vit  # noqa: F401
import src.models.vit.vit  # noqa: F401
from src.client.afad_client import AFADClient
from src.client.fedgen_client import FedGenClient
from src.client.heterofl_client import HeteroFLClient
from src.data.dataset_config import get_dataset_config
from src.data.mnist_loader import load_mnist_data
from src.models.fedgen_wrapper import FedGenModelWrapper
from src.models.registry import ModelRegistry
from src.server.generator.fedgen_generator import FedGenGenerator
from src.server.strategy.afad_strategy import AFADStrategy
from src.utils.logger import setup_logger

logger = setup_logger("Comparison")

# Phase 3: 10 clients, 2 families with width-scaled sub-models
CID_TO_MODEL_P3 = {
    "0": "heterofl_resnet18",
    "1": "heterofl_resnet18",
    "2": "heterofl_resnet18",
    "3": "heterofl_resnet18",
    "4": "heterofl_resnet18",
    "5": "heterofl_vit_small",
    "6": "heterofl_vit_small",
    "7": "heterofl_vit_small",
    "8": "heterofl_vit_small",
    "9": "heterofl_vit_small",
}
CID_TO_FAMILY_P3 = {
    "0": "cnn",
    "1": "cnn",
    "2": "cnn",
    "3": "cnn",
    "4": "cnn",
    "5": "vit",
    "6": "vit",
    "7": "vit",
    "8": "vit",
    "9": "vit",
}
CID_TO_RATE_P3 = {
    "0": 1.0,
    "1": 1.0,
    "2": 0.5,
    "3": 0.5,
    "4": 0.25,
    "5": 1.0,
    "6": 1.0,
    "7": 0.5,
    "8": 0.5,
    "9": 0.25,
}
FAMILY_MODEL_NAMES_P3 = {
    "cnn": "heterofl_resnet18",
    "vit": "heterofl_vit_small",
}

LATENT_DIM = 32

device_type = "cuda" if torch.cuda.is_available() else "cpu"
num_gpus = 0.15 if device_type == "cuda" else 0.0
CLIENT_RESOURCES = {"num_cpus": 1.0, "num_gpus": num_gpus}


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_plain_model_factories(
    cid_to_model: dict[str, str],
) -> dict[str, callable]:
    """Build factories that create plain (unwrapped) models."""
    model_names = set(cid_to_model.values())
    factories = {}
    for name in model_names:
        factories[name] = lambda num_classes=10, _n=name: ModelRegistry.create_model(
            _n, num_classes=num_classes
        )
    return factories


def build_wrapped_model_factories(
    family_model_names: dict[str, str],
    latent_dim: int = LATENT_DIM,
) -> dict[str, callable]:
    """Build factories that create FedGenModelWrapper-wrapped models.

    For AFAD/FedGen modes, the server needs to reconstruct wrapped models
    with forward_from_latent support for generator training.
    """
    factories = {}
    for _family, name in family_model_names.items():
        if name not in factories:
            factories[name] = lambda num_classes=10, _n=name, _ld=latent_dim: (
                FedGenModelWrapper(
                    ModelRegistry.create_model(_n, num_classes=num_classes),
                    latent_dim=_ld,
                    num_classes=num_classes,
                )
            )
    return factories


def heterofl_client_fn_builder(
    train_loaders,
    test_loader,
    cid_to_model: dict[str, str],
    cid_to_family: dict[str, str],
    num_classes: int,
    local_epochs: int,
    cid_to_rate: dict[str, float],
):
    """Build client_fn for HeteroFL Only (plain models, no KD)."""

    def client_fn(cid: str) -> fl.client.Client:
        import src.models.cnn.heterofl_resnet  # noqa: F401
        import src.models.vit.heterofl_vit  # noqa: F401

        model_name = cid_to_model.get(cid, "heterofl_resnet18")
        family = cid_to_family.get(cid, "cnn")
        model_rate = cid_to_rate.get(cid, 1.0)
        device = get_device()
        model = ModelRegistry.create_model(
            model_name, num_classes=num_classes, model_rate=model_rate
        )
        train_loader = train_loaders[int(cid) % len(train_loaders)]

        return HeteroFLClient(
            cid=cid,
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            epochs=local_epochs,
            device=device,
            family=family,
            model_rate=model_rate,
            model_name=model_name,
            num_classes=num_classes,
        ).to_client()

    return client_fn


def fedgen_client_fn_builder(
    train_loaders,
    test_loader,
    cid_to_model: dict[str, str],
    cid_to_family: dict[str, str],
    num_classes: int,
    local_epochs: int,
):
    """Build client_fn for FedGen Only (wrapped models at rate=1.0, KD)."""

    def client_fn(cid: str) -> fl.client.Client:
        import src.models.cnn.heterofl_resnet  # noqa: F401
        import src.models.vit.heterofl_vit  # noqa: F401

        model_name = cid_to_model.get(cid, "heterofl_resnet18")
        device = get_device()
        base_model = ModelRegistry.create_model(model_name, num_classes=num_classes)
        model = FedGenModelWrapper(
            base_model, latent_dim=LATENT_DIM, num_classes=num_classes
        )
        generator = FedGenGenerator(
            noise_dim=LATENT_DIM,
            num_classes=num_classes,
            latent_dim=LATENT_DIM,
        )
        train_loader = train_loaders[int(cid) % len(train_loaders)]

        return FedGenClient(
            cid=cid,
            model=model,
            generator=generator,
            train_loader=train_loader,
            epochs=local_epochs,
            device=device,
            num_classes=num_classes,
        ).to_client()

    return client_fn


def afad_client_fn_builder(
    train_loaders,
    test_loader,
    cid_to_model: dict[str, str],
    cid_to_family: dict[str, str],
    num_classes: int,
    local_epochs: int,
    cid_to_rate: dict[str, float],
):
    """Build client_fn for AFAD Hybrid (wrapped + width-scaled, KD)."""

    def client_fn(cid: str) -> fl.client.Client:
        import src.models.cnn.heterofl_resnet  # noqa: F401
        import src.models.vit.heterofl_vit  # noqa: F401

        model_name = cid_to_model.get(cid, "heterofl_resnet18")
        family = cid_to_family.get(cid, "cnn")
        model_rate = cid_to_rate.get(cid, 1.0)
        device = get_device()
        base_model = ModelRegistry.create_model(
            model_name, num_classes=num_classes, model_rate=model_rate
        )
        model = FedGenModelWrapper(
            base_model, latent_dim=LATENT_DIM, num_classes=num_classes
        )
        generator = FedGenGenerator(
            noise_dim=LATENT_DIM,
            num_classes=num_classes,
            latent_dim=LATENT_DIM,
        )
        train_loader = train_loaders[int(cid) % len(train_loaders)]

        # Rate-dependent KD scaling: sub-rate clients get stronger guidance
        # alpha = 0.5 / rate: rate=1.0->0.5, rate=0.5->1.0, rate=0.25->2.0
        kd_scale = 0.5 / model_rate
        return AFADClient(
            cid=cid,
            model=model,
            generator=generator,
            train_loader=train_loader,
            val_loader=test_loader,
            epochs=local_epochs,
            device=device,
            family=family,
            model_rate=model_rate,
            model_name=model_name,
            num_classes=num_classes,
            generative_alpha=kd_scale,
            generative_beta=kd_scale,
        ).to_client()

    return client_fn


def evaluate_metrics_aggregation_fn(
    eval_metrics: list[tuple[int, dict]],
) -> dict:
    total = sum(num for num, _ in eval_metrics)
    if total == 0:
        return {}
    accuracy = sum(num * m.get("accuracy", 0.0) for num, m in eval_metrics) / total
    return {"accuracy": accuracy}


def _build_strategy(
    enable_fedgen: bool,
    enable_heterofl: bool,
    num_classes: int,
    num_clients: int,
    num_rounds: int,
    cid_to_family: dict[str, str],
    cid_to_rate: dict[str, float] | None,
    family_model_names: dict[str, str],
) -> AFADStrategy:
    """Build AFADStrategy with appropriate configuration."""
    device = get_device()

    # Generator (None for HeteroFL Only)
    generator = None
    if enable_fedgen:
        generator = FedGenGenerator(
            noise_dim=LATENT_DIM,
            num_classes=num_classes,
            latent_dim=LATENT_DIM,
        )

    # Model factories
    if enable_fedgen:
        model_factories = build_wrapped_model_factories(
            family_model_names, latent_dim=LATENT_DIM
        )
    else:
        model_factories = build_plain_model_factories(
            {
                str(i): name
                for i, name in enumerate(
                    [family_model_names[f] for f in sorted(family_model_names)]
                )
            }
        )

    # Initial params (just for Flower framework, not actually used)
    first_model_name = next(iter(family_model_names.values()))
    if enable_fedgen:
        initial_model = FedGenModelWrapper(
            ModelRegistry.create_model(first_model_name, num_classes=num_classes),
            latent_dim=LATENT_DIM,
            num_classes=num_classes,
        )
    else:
        initial_model = ModelRegistry.create_model(
            first_model_name, num_classes=num_classes
        )
    initial_params = fl.common.ndarrays_to_parameters(
        [val.cpu().numpy() for val in initial_model.state_dict().values()]
    )

    fedgen_config = {
        "gen_lr": 3e-4,
        "batch_size": 128,
        "ensemble_alpha": 1.0,
        "ensemble_eta": 1.0,
        "gen_epochs": 2,
        "teacher_iters": 25,
        "device": device,
    }

    training_config = {
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "local_epochs": 1,
        "fedprox_mu": 0.0,
    }

    client_model_rates = cid_to_rate if enable_heterofl else None

    strategy = AFADStrategy(
        initial_parameters=initial_params,
        generator=generator,
        model_factories=model_factories,
        client_model_rates=client_model_rates,
        family_model_names=family_model_names,
        fedgen_config=fedgen_config,
        training_config=training_config,
        enable_fedgen=enable_fedgen,
        enable_heterofl=enable_heterofl,
        num_rounds=num_rounds,
        num_classes=num_classes,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_evaluate_clients=num_clients,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    # Pre-register client families
    for cid, family in cid_to_family.items():
        strategy.set_client_family(cid, family)

    return strategy


def run_single_experiment(
    label: str,
    train_loaders,
    test_loader,
    enable_fedgen: bool,
    enable_heterofl: bool,
    num_classes: int,
    num_clients: int,
    num_rounds: int,
    local_epochs: int,
    seed: int = 42,
    cid_to_model: dict[str, str] | None = None,
    cid_to_family: dict[str, str] | None = None,
    cid_to_rate: dict[str, float] | None = None,
    family_model_names: dict[str, str] | None = None,
) -> list[dict]:
    """Run one Flower simulation and return per-round metrics."""
    cid_to_model = cid_to_model or CID_TO_MODEL_P3
    cid_to_family = cid_to_family or CID_TO_FAMILY_P3
    cid_to_rate = cid_to_rate or CID_TO_RATE_P3
    family_model_names = family_model_names or FAMILY_MODEL_NAMES_P3

    logger.info(f"{'=' * 60}")
    logger.info(f"Starting experiment: {label}")
    logger.info(
        f"  enable_fedgen={enable_fedgen}, enable_heterofl={enable_heterofl}, "
        f"clients={num_clients}"
    )
    logger.info(f"{'=' * 60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    strategy = _build_strategy(
        enable_fedgen=enable_fedgen,
        enable_heterofl=enable_heterofl,
        num_classes=num_classes,
        num_clients=num_clients,
        num_rounds=num_rounds,
        cid_to_family=cid_to_family,
        cid_to_rate=cid_to_rate,
        family_model_names=family_model_names,
    )

    # Build client_fn based on experiment mode
    if enable_heterofl and enable_fedgen:
        # AFAD Hybrid
        client_fn = afad_client_fn_builder(
            train_loaders,
            test_loader,
            cid_to_model,
            cid_to_family,
            num_classes,
            local_epochs,
            cid_to_rate,
        )
    elif enable_fedgen:
        # FedGen Only (rate=1.0 for all)
        client_fn = fedgen_client_fn_builder(
            train_loaders,
            test_loader,
            cid_to_model,
            cid_to_family,
            num_classes,
            local_epochs,
        )
    else:
        # HeteroFL Only
        client_fn = heterofl_client_fn_builder(
            train_loaders,
            test_loader,
            cid_to_model,
            cid_to_family,
            num_classes,
            local_epochs,
            cid_to_rate,
        )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=CLIENT_RESOURCES,
        ray_init_args={
            "log_to_driver": False,
            "object_store_memory": 500_000_000,  # 500MB: avoids /dev/shm exhaustion on WSL2
            "runtime_env": {},  # skip working-dir package upload (prevents startup hang)
        },
    )

    if ray.is_initialized():
        ray.shutdown()

    rounds = []
    for rm in strategy.metrics_collector.rounds:
        rounds.append(
            {
                "round": rm.round_num,
                "accuracy": rm.accuracy,
                "loss": rm.loss,
                "wall_time": rm.wall_time,
            }
        )

    summary = strategy.metrics_collector.summary()
    logger.info(
        f"{label} finished: best_accuracy={summary.get('best_accuracy', 0):.4f}"
    )

    return rounds


def print_comparison_table(results: dict[str, list[dict]]) -> None:
    """Print side-by-side comparison of all methods."""
    labels = list(results.keys())
    num_rounds = max(len(v) for v in results.values())

    header = f"{'Round':>5}"
    for label in labels:
        header += f" | {label:>22}"
    separator = "-" * len(header)

    print("\n")
    print("=" * len(header))
    print("  COMPARISON: Accuracy per Round")
    print("=" * len(header))
    print(header)
    print(separator)

    for i in range(num_rounds):
        row = f"{i + 1:>5}"
        for label in labels:
            if i < len(results[label]):
                acc = results[label][i]["accuracy"]
                row += f" | {acc:>21.2%}"
            else:
                row += f" | {'N/A':>21}"
        print(row)

    print(separator)
    row = f"{'BEST':>5}"
    for label in labels:
        best = max(r["accuracy"] for r in results[label])
        row += f" | {best:>21.2%}"
    print(row)

    row = f"{'FINAL':>5}"
    for label in labels:
        final = results[label][-1]["accuracy"]
        row += f" | {final:>21.2%}"
    print(row)

    print("\n")
    for label in labels:
        total_time = sum(r["wall_time"] for r in results[label])
        print(f"  {label}: total_time={total_time:.1f}s")


def main():
    from src.utils.config_loader import load_config

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/afad_config.yaml"
    config_dict = load_config(config_path)

    dataset_name = config_dict["data"].get("dataset", "mnist")
    ds_cfg = get_dataset_config(dataset_name)
    num_classes = ds_cfg.num_classes
    num_clients = config_dict["server"]["min_clients"]
    num_rounds = config_dict["experiment"].get("num_rounds", 40)
    local_epochs = config_dict.get("training", {}).get("local_epochs", 3)
    batch_size = config_dict["data"].get("batch_size", 64)
    seed = config_dict["experiment"].get("seed", 42)

    logger.info(f"Loading {dataset_name} data for {num_clients} clients...")

    if dataset_name == "organamnist":
        from src.data.medmnist_loader import load_organamnist_data

        data_cfg = config_dict["data"]
        train_loaders, test_loader = load_organamnist_data(
            num_clients=num_clients,
            batch_size=batch_size,
            alpha=data_cfg.get("dirichlet_alpha", 0.5),
            distribution=data_cfg.get("distribution", "non_iid"),
            seed=seed,
        )
    else:
        train_loaders, test_loader = load_mnist_data(
            num_clients=num_clients,
            batch_size=batch_size,
        )

    exp_kwargs = {
        "train_loaders": train_loaders,
        "test_loader": test_loader,
        "num_classes": num_classes,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "seed": seed,
        "cid_to_model": CID_TO_MODEL_P3,
        "cid_to_family": CID_TO_FAMILY_P3,
        "cid_to_rate": CID_TO_RATE_P3,
        "family_model_names": FAMILY_MODEL_NAMES_P3,
    }

    results: dict[str, list[dict]] = {}

    # --- Experiment 1: HeteroFL Only ---
    results["HeteroFL Only"] = run_single_experiment(
        label="HeteroFL Only",
        enable_fedgen=False,
        enable_heterofl=True,
        **exp_kwargs,
    )

    # --- Experiment 2: FedGen Only ---
    results["FedGen Only"] = run_single_experiment(
        label="FedGen Only",
        enable_fedgen=True,
        enable_heterofl=False,
        **exp_kwargs,
    )

    # --- Experiment 3: AFAD Hybrid ---
    results["AFAD Hybrid"] = run_single_experiment(
        label="AFAD Hybrid",
        enable_fedgen=True,
        enable_heterofl=True,
        **exp_kwargs,
    )

    print_comparison_table(results)


if __name__ == "__main__":
    main()
