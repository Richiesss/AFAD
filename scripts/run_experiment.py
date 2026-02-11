import os
import sys

import flwr as fl
import torch

# Add project root to path
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

logger = setup_logger("AFADExperiment")

# Check for GPU availability
device_type = "cuda" if torch.cuda.is_available() else "cpu"
num_gpus = 0.15 if device_type == "cuda" else 0.0
CLIENT_RESOURCES = {"num_cpus": 1.0, "num_gpus": num_gpus}

# Phase 1: 5 clients, 5 unique architectures
CID_TO_MODEL_5 = {
    "0": "resnet50",
    "1": "mobilenetv3_large",
    "2": "resnet18",
    "3": "vit_tiny",
    "4": "deit_small",
}

CID_TO_FAMILY_5 = {
    "0": "cnn",
    "1": "cnn",
    "2": "cnn",
    "3": "vit",
    "4": "vit",
}

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


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_client_mappings(
    num_clients: int,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return (cid_to_model, cid_to_family) for the given client count."""
    if num_clients <= 5:
        return CID_TO_MODEL_5, CID_TO_FAMILY_5
    return CID_TO_MODEL_10, CID_TO_FAMILY_10


def build_model_factories(
    cid_to_model: dict[str, str],
) -> dict[str, callable]:
    """Build factory functions for each model used in the experiment."""
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
    config,
    cid_to_model: dict[str, str],
    cid_to_family: dict[str, str],
    num_classes: int,
):
    """Closure to create client_fn with access to data loaders."""

    def client_fn(cid: str) -> fl.client.Client:
        # Import models inside Ray worker to ensure registration
        import src.models.cnn.mobilenet  # noqa: F401
        import src.models.cnn.resnet  # noqa: F401
        import src.models.vit.deit  # noqa: F401
        import src.models.vit.vit  # noqa: F401

        model_name = cid_to_model.get(cid, "resnet18")
        family = cid_to_family.get(cid, "cnn")
        device = get_device()

        model = ModelRegistry.create_model(model_name, num_classes=num_classes)

        client_id_int = int(cid)
        train_loader = train_loaders[client_id_int % len(train_loaders)]

        training_cfg = config.get("training", {})

        return AFADClient(
            cid=cid,
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            epochs=training_cfg.get("local_epochs", 1),
            device=device,
            family=family,
            model_name=model_name,
            lr=training_cfg.get("learning_rate", 0.01),
            momentum=training_cfg.get("momentum", 0.9),
            weight_decay=training_cfg.get("weight_decay", 0.0001),
            num_classes=num_classes,
        ).to_client()

    return client_fn


def evaluate_metrics_aggregation_fn(
    eval_metrics: list[tuple[int, dict]],
) -> dict:
    """Aggregate evaluation metrics from all clients."""
    total = sum(num for num, _ in eval_metrics)
    if total == 0:
        return {}
    accuracy = sum(num * m.get("accuracy", 0.0) for num, m in eval_metrics) / total
    return {"accuracy": accuracy}


def main():
    from src.utils.config_loader import load_config

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/afad_config.yaml"
    config_dict = load_config(config_path)

    # Dataset config
    dataset_name = config_dict["data"].get("dataset", "mnist")
    ds_cfg = get_dataset_config(dataset_name)
    num_classes = ds_cfg.num_classes
    num_clients = config_dict["server"]["min_clients"]

    # Client mappings
    cid_to_model, cid_to_family = get_client_mappings(num_clients)

    # Data loading
    if dataset_name == "organamnist":
        from src.data.medmnist_loader import load_organamnist_data

        data_cfg = config_dict["data"]
        train_loaders, test_loader = load_organamnist_data(
            num_clients=num_clients,
            batch_size=data_cfg.get("batch_size", 64),
            alpha=data_cfg.get("dirichlet_alpha", 0.5),
            distribution=data_cfg.get("distribution", "non_iid"),
        )
    else:
        train_loaders, test_loader = load_mnist_data(
            num_clients=num_clients,
            batch_size=config_dict["data"]["batch_size"],
        )

    # Strategy components
    family_router = FamilyRouter()
    strategy_cfg = config_dict.get("strategy", {})
    fedgen_yaml = strategy_cfg.get("fedgen", {})

    generator = SyntheticGenerator(
        noise_dim=32,
        num_classes=num_classes,
        hidden_dim=256,
        mean=ds_cfg.mean[0],
        std=ds_cfg.std[0],
    )

    # Initial parameters (dummy for Flower)
    initial_model = ModelRegistry.create_model("resnet18", num_classes=num_classes)
    initial_params = fl.common.ndarrays_to_parameters(
        [val.cpu().numpy() for val in initial_model.state_dict().values()]
    )

    # Model factories for FedGen reconstruction
    model_factories = build_model_factories(cid_to_model)

    # FedGen config
    fedgen_config = {
        "gen_lr": fedgen_yaml.get("gen_lr", 3e-4),
        "batch_size": fedgen_yaml.get("batch_size", 128),
        "ensemble_alpha": fedgen_yaml.get("ensemble_alpha", 1.0),
        "ensemble_eta": fedgen_yaml.get("ensemble_eta", 1.0),
        "gen_epochs": fedgen_yaml.get("gen_epochs", 2),
        "teacher_iters": fedgen_yaml.get("teacher_iters", 25),
        "temperature": fedgen_yaml.get("temperature", 4.0),
        "distill_lr": fedgen_yaml.get("distill_lr", 1e-4),
        "distill_epochs": fedgen_yaml.get("distill_epochs", 1),
        "distill_steps": fedgen_yaml.get("distill_steps", 5),
        "distill_alpha": fedgen_yaml.get("distill_alpha", 1.0),
        "distill_beta": fedgen_yaml.get("distill_beta", 0.1),
        "distill_every": fedgen_yaml.get("distill_every", 2),
        "device": get_device(),
    }

    # Training config to propagate to clients
    training_cfg = config_dict.get("training", {})
    training_config = {
        "lr": training_cfg.get("learning_rate", 0.01),
        "momentum": training_cfg.get("momentum", 0.9),
        "weight_decay": training_cfg.get("weight_decay", 0.0001),
        "local_epochs": training_cfg.get("local_epochs", 1),
    }

    num_rounds = config_dict["experiment"]["num_rounds"]

    strategy = AFADStrategy(
        initial_parameters=initial_params,
        initial_generator=generator,
        family_router=family_router,
        model_factories=model_factories,
        fedgen_config=fedgen_config,
        training_config=training_config,
        num_rounds=num_rounds,
        num_classes=num_classes,
        min_fit_clients=config_dict["server"]["min_fit_clients"],
        min_available_clients=config_dict["server"]["min_clients"],
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_evaluate_clients=config_dict["server"]["min_clients"],
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    # Simulation
    logger.info(
        f"Starting AFAD simulation: {num_rounds} rounds, "
        f"dataset={dataset_name}, clients={num_clients}"
    )

    fl.simulation.start_simulation(
        client_fn=client_fn_builder(
            train_loaders,
            test_loader,
            config_dict,
            cid_to_model,
            cid_to_family,
            num_classes,
        ),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=CLIENT_RESOURCES,
    )

    # Print final summary
    summary = strategy.metrics_collector.summary()
    if summary:
        logger.info("=" * 50)
        logger.info("Experiment Summary:")
        for key, val in summary.items():
            logger.info(f"  {key}: {val:.4f}")
        logger.info("=" * 50)


if __name__ == "__main__":
    main()
