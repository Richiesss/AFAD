"""
Compare HeteroFL-only, FedGen-only, and AFAD hybrid (HeteroFL + FedGen).

Runs 3 Flower simulations sequentially with the same data split and
collects per-round accuracy/loss for each method.
"""

import os
import sys

import flwr as fl
import ray
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.models.cnn.mobilenet  # noqa: F401 - register models
import src.models.cnn.resnet  # noqa: F401
import src.models.vit.deit  # noqa: F401
import src.models.vit.vit  # noqa: F401
from src.client.afad_client import AFADClient
from src.data.mnist_loader import load_mnist_data
from src.models.registry import ModelRegistry
from src.routing.family_router import FamilyRouter
from src.server.generator.synthetic_generator import SyntheticGenerator
from src.server.strategy.afad_strategy import AFADStrategy
from src.utils.logger import setup_logger

logger = setup_logger("Comparison")

# --- Experiment constants ---
NUM_CLIENTS = 5
NUM_ROUNDS = 20
BATCH_SIZE = 64
LOCAL_EPOCHS = 2
SEED = 42

CID_TO_MODEL = {
    "0": "resnet50",
    "1": "mobilenetv3_large",
    "2": "resnet18",
    "3": "vit_tiny",
    "4": "deit_small",
}
CID_TO_FAMILY = {"0": "cnn", "1": "cnn", "2": "cnn", "3": "vit", "4": "vit"}

device_type = "cuda" if torch.cuda.is_available() else "cpu"
num_gpus = 0.15 if device_type == "cuda" else 0.0
CLIENT_RESOURCES = {"num_cpus": 1.0, "num_gpus": num_gpus}


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model_factories() -> dict[str, callable]:
    model_names = set(CID_TO_MODEL.values())
    factories = {}
    for name in model_names:
        factories[name] = lambda num_classes=10, _n=name: ModelRegistry.create_model(
            _n, num_classes=num_classes
        )
    return factories


def client_fn_builder(train_loaders, test_loader):
    def client_fn(cid: str) -> fl.client.Client:
        import src.models.cnn.mobilenet  # noqa: F401
        import src.models.cnn.resnet  # noqa: F401
        import src.models.vit.deit  # noqa: F401
        import src.models.vit.vit  # noqa: F401

        model_name = CID_TO_MODEL.get(cid, "resnet18")
        family = CID_TO_FAMILY.get(cid, "cnn")
        device = get_device()
        model = ModelRegistry.create_model(model_name, num_classes=10)
        train_loader = train_loaders[int(cid) % len(train_loaders)]

        return AFADClient(
            cid=cid,
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            epochs=LOCAL_EPOCHS,
            device=device,
            family=family,
            model_name=model_name,
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
) -> list[dict]:
    """Run one Flower simulation and return per-round metrics."""
    logger.info(f"{'=' * 60}")
    logger.info(f"Starting experiment: {label}")
    logger.info(f"  enable_fedgen={enable_fedgen}")
    logger.info(f"{'=' * 60}")

    torch.manual_seed(SEED)

    family_router = FamilyRouter()
    generator = SyntheticGenerator(noise_dim=32, num_classes=10, hidden_dim=256)
    model_factories = build_model_factories()

    initial_model = ModelRegistry.create_model("resnet18", num_classes=10)
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
        "device": "cpu",  # Server-side generator/distillation on CPU (GPU reserved for Ray actors)
    }

    training_config = {
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "local_epochs": LOCAL_EPOCHS,
    }

    strategy = AFADStrategy(
        initial_parameters=initial_params,
        initial_generator=generator,
        family_router=family_router,
        model_factories=model_factories,
        fedgen_config=fedgen_config,
        training_config=training_config,
        enable_fedgen=enable_fedgen,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_evaluate_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    client_fn = client_fn_builder(train_loaders, test_loader)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
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
    logger.info("Loading MNIST data...")
    train_loaders, test_loader = load_mnist_data(
        num_clients=NUM_CLIENTS,
        batch_size=BATCH_SIZE,
    )

    # Store data loaders so all experiments use the same data split
    results: dict[str, list[dict]] = {}

    # --- Experiment 1: HeteroFL Only (no FedGen) ---
    results["HeteroFL Only"] = run_single_experiment(
        label="HeteroFL Only",
        train_loaders=train_loaders,
        test_loader=test_loader,
        enable_fedgen=False,
    )

    # --- Experiment 2: FedGen Only ---
    # Note: With 5 unique architectures, HeteroFL aggregation is a no-op
    # (1 client per signature group = simple copy). Therefore FedGen-only
    # produces equivalent results to the AFAD hybrid in this configuration.
    results["FedGen Only"] = run_single_experiment(
        label="FedGen Only",
        train_loaders=train_loaders,
        test_loader=test_loader,
        enable_fedgen=True,
    )

    # --- Experiment 3: AFAD Hybrid (HeteroFL + FedGen) ---
    # In this 5-unique-architecture setup, this is functionally identical
    # to FedGen Only, since HeteroFL degenerates to identity.
    # Included for completeness and to demonstrate the equivalence.
    results["AFAD Hybrid"] = run_single_experiment(
        label="AFAD Hybrid",
        train_loaders=train_loaders,
        test_loader=test_loader,
        enable_fedgen=True,
    )

    # --- Print comparison ---
    print_comparison_table(results)


if __name__ == "__main__":
    main()
