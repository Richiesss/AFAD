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

import src.models.cnn.mobilenet  # noqa: F401 - register models
import src.models.cnn.resnet  # noqa: F401
import src.models.vit.deit  # noqa: F401
import src.models.vit.vit  # noqa: F401
from src.client.afad_client import AFADClient
from src.data.dataset_config import get_dataset_config
from src.data.mnist_loader import load_mnist_data
from src.models.registry import ModelRegistry
from src.routing.family_router import FamilyRouter
from src.server.generator.synthetic_generator import SyntheticGenerator
from src.server.strategy.afad_strategy import AFADStrategy
from src.utils.logger import setup_logger

logger = setup_logger("Comparison")

# Phase 1: 5 clients, 5 unique architectures
CID_TO_MODEL_5 = {
    "0": "resnet50",
    "1": "mobilenetv3_large",
    "2": "resnet18",
    "3": "vit_tiny",
    "4": "deit_small",
}
CID_TO_FAMILY_5 = {"0": "cnn", "1": "cnn", "2": "cnn", "3": "vit", "4": "vit"}

# Phase 2: 10 clients, 5 architectures x 2
CID_TO_MODEL_10 = {
    "0": "resnet50",
    "1": "resnet50",
    "2": "mobilenetv3_large",
    "3": "mobilenetv3_large",
    "4": "resnet18",
    "5": "resnet18",
    "6": "vit_tiny",
    "7": "vit_tiny",
    "8": "deit_small",
    "9": "deit_small",
}
CID_TO_FAMILY_10 = {
    "0": "cnn",
    "1": "cnn",
    "2": "cnn",
    "3": "cnn",
    "4": "cnn",
    "5": "cnn",
    "6": "vit",
    "7": "vit",
    "8": "vit",
    "9": "vit",
}

device_type = "cuda" if torch.cuda.is_available() else "cpu"
num_gpus = 0.15 if device_type == "cuda" else 0.0
CLIENT_RESOURCES = {"num_cpus": 1.0, "num_gpus": num_gpus}


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_client_mappings(
    num_clients: int,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return (cid_to_model, cid_to_family) for the given client count."""
    if num_clients <= 5:
        return CID_TO_MODEL_5, CID_TO_FAMILY_5
    return CID_TO_MODEL_10, CID_TO_FAMILY_10


def build_model_factories(cid_to_model: dict[str, str]) -> dict[str, callable]:
    model_names = set(cid_to_model.values())
    factories = {}
    for name in model_names:
        factories[name] = lambda num_classes=10, _n=name: ModelRegistry.create_model(
            _n, num_classes=num_classes
        )
    return factories


def client_fn_builder(
    train_loaders,
    test_loader,
    cid_to_model: dict[str, str],
    cid_to_family: dict[str, str],
    num_classes: int,
    local_epochs: int,
):
    def client_fn(cid: str) -> fl.client.Client:
        import src.models.cnn.mobilenet  # noqa: F401
        import src.models.cnn.resnet  # noqa: F401
        import src.models.vit.deit  # noqa: F401
        import src.models.vit.vit  # noqa: F401

        model_name = cid_to_model.get(cid, "resnet18")
        family = cid_to_family.get(cid, "cnn")
        device = get_device()
        model = ModelRegistry.create_model(model_name, num_classes=num_classes)
        train_loader = train_loaders[int(cid) % len(train_loaders)]

        return AFADClient(
            cid=cid,
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            epochs=local_epochs,
            device=device,
            family=family,
            model_name=model_name,
            num_classes=num_classes,
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


def run_single_experiment(
    label: str,
    train_loaders,
    test_loader,
    enable_fedgen: bool,
    cid_to_model: dict[str, str],
    cid_to_family: dict[str, str],
    num_classes: int,
    num_clients: int,
    num_rounds: int,
    local_epochs: int,
    ds_mean: float,
    ds_std: float,
    seed: int = 42,
    fedprox_mu: float = 0.0,
) -> list[dict]:
    """Run one Flower simulation and return per-round metrics."""
    logger.info(f"{'=' * 60}")
    logger.info(f"Starting experiment: {label}")
    logger.info(f"  enable_fedgen={enable_fedgen}, clients={num_clients}")
    logger.info(f"{'=' * 60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    family_router = FamilyRouter()
    generator = SyntheticGenerator(
        noise_dim=32,
        num_classes=num_classes,
        hidden_dim=256,
        mean=ds_mean,
        std=ds_std,
    )
    model_factories = build_model_factories(cid_to_model)

    initial_model = ModelRegistry.create_model("resnet18", num_classes=num_classes)
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
        "temperature": 4.0,
        "distill_lr": 1e-4,
        "distill_epochs": 1,
        "distill_steps": 5,
        "distill_alpha": 1.0,
        "distill_beta": 0.1,
        "distill_every": 2,
        "device": get_device(),
    }

    training_config = {
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "local_epochs": local_epochs,
        "fedprox_mu": fedprox_mu,
    }

    strategy = AFADStrategy(
        initial_parameters=initial_params,
        initial_generator=generator,
        family_router=family_router,
        model_factories=model_factories,
        fedgen_config=fedgen_config,
        training_config=training_config,
        enable_fedgen=enable_fedgen,
        num_rounds=num_rounds,
        num_classes=num_classes,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_evaluate_clients=num_clients,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    client_fn = client_fn_builder(
        train_loaders,
        test_loader,
        cid_to_model,
        cid_to_family,
        num_classes,
        local_epochs,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=CLIENT_RESOURCES,
    )

    # Ensure Ray is shut down before next experiment
    if ray.is_initialized():
        ray.shutdown()

    # Collect per-round metrics
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

    # Header
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

    # Summary row
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

    # Loss comparison
    print("\n")
    print("=" * len(header))
    print("  COMPARISON: Loss per Round")
    print("=" * len(header))
    print(header)
    print(separator)

    for i in range(num_rounds):
        row = f"{i + 1:>5}"
        for label in labels:
            if i < len(results[label]):
                loss = results[label][i]["loss"]
                row += f" | {loss:>21.4f}"
            else:
                row += f" | {'N/A':>21}"
        print(row)

    print(separator)
    row = f"{'FINAL':>5}"
    for label in labels:
        final = results[label][-1]["loss"]
        row += f" | {final:>21.4f}"
    print(row)

    # Wall time
    print("\n")
    for label in labels:
        total_time = sum(r["wall_time"] for r in results[label])
        print(f"  {label}: total_time={total_time:.1f}s")


def main():
    from src.utils.config_loader import load_config

    # Load config (default: Phase 1)
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/afad_config.yaml"
    config_dict = load_config(config_path)

    # Dataset
    dataset_name = config_dict["data"].get("dataset", "mnist")
    ds_cfg = get_dataset_config(dataset_name)
    num_classes = ds_cfg.num_classes
    num_clients = config_dict["server"]["min_clients"]
    num_rounds = config_dict["experiment"].get("num_rounds", 40)
    local_epochs = config_dict.get("training", {}).get("local_epochs", 3)
    batch_size = config_dict["data"].get("batch_size", 64)
    seed = config_dict["experiment"].get("seed", 42)
    fedprox_mu = config_dict.get("training", {}).get("fedprox_mu", 0.0)

    # Client mappings
    cid_to_model, cid_to_family = get_client_mappings(num_clients)

    logger.info(f"Loading {dataset_name} data for {num_clients} clients...")

    # Data loading
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

    # Common experiment kwargs
    exp_kwargs = {
        "train_loaders": train_loaders,
        "test_loader": test_loader,
        "cid_to_model": cid_to_model,
        "cid_to_family": cid_to_family,
        "num_classes": num_classes,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "ds_mean": ds_cfg.mean[0],
        "ds_std": ds_cfg.std[0],
        "seed": seed,
        "fedprox_mu": fedprox_mu,
    }

    results: dict[str, list[dict]] = {}

    # --- Experiment 1: HeteroFL Only (no FedGen) ---
    results["HeteroFL Only"] = run_single_experiment(
        label="HeteroFL Only",
        enable_fedgen=False,
        **exp_kwargs,
    )

    # --- Experiment 2: FedGen Only ---
    results["FedGen Only"] = run_single_experiment(
        label="FedGen Only",
        enable_fedgen=True,
        **exp_kwargs,
    )

    # --- Experiment 3: AFAD Hybrid (HeteroFL + FedGen) ---
    results["AFAD Hybrid"] = run_single_experiment(
        label="AFAD Hybrid",
        enable_fedgen=True,
        **exp_kwargs,
    )

    # --- Print comparison ---
    print_comparison_table(results)


if __name__ == "__main__":
    main()
