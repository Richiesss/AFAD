"""Distributed FL server for AFAD.

Usage:
    uv run python scripts/run_distributed.py server --config config/afad_config.yaml --port 8080
    uv run python scripts/run_distributed.py client --config config/afad_config.yaml --server-address 100.118.49.59:8080 --client-id 5
"""

import argparse
import os
import sys

import flwr as fl
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.models.cnn.heterofl_resnet  # noqa: F401
import src.models.vit.heterofl_vit  # noqa: F401
from src.client.afad_client import AFADClient
from src.client.fedgen_client import FedGenClient
from src.client.heterofl_client import HeteroFLClient
from src.data.dataset_config import get_dataset_config
from src.models.fedgen_wrapper import FedGenModelWrapper
from src.models.registry import ModelRegistry
from src.server.generator.fedgen_generator import FedGenGenerator
from src.server.strategy.afad_strategy import AFADStrategy
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger("Distributed")
LATENT_DIM = 32

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def _parse_clients_from_config(config_dict):
    clients = config_dict.get("clients", [])
    cid_to_model, cid_to_family, cid_to_rate, family_model_names = {}, {}, {}, {}
    for c in clients:
        cid = str(c["id"])
        model = c.get("model", "heterofl_resnet18")
        family = c.get("family", "cnn")
        cid_to_model[cid] = model
        cid_to_family[cid] = family
        cid_to_rate[cid] = float(c.get("model_rate", 1.0))
        family_model_names.setdefault(family, model)
    return cid_to_model, cid_to_family, cid_to_rate, family_model_names

def build_wrapped_model_factories(family_model_names, latent_dim=LATENT_DIM):
    factories = {}
    for _fam, name in family_model_names.items():
        if name not in factories:
            factories[name] = lambda nc=10, _n=name, _ld=latent_dim: FedGenModelWrapper(ModelRegistry.create_model(_n, num_classes=nc), latent_dim=_ld, num_classes=nc)
    return factories

def build_plain_model_factories(cid_to_model):
    factories = {}
    for name in set(cid_to_model.values()):
        factories[name] = lambda nc=10, _n=name: ModelRegistry.create_model(_n, num_classes=nc)
    return factories

def evaluate_metrics_aggregation_fn(eval_metrics):
    total = sum(n for n, _ in eval_metrics)
    if total == 0: return {}
    return {"accuracy": sum(n * m.get("accuracy", 0.0) for n, m in eval_metrics) / total}

def run_server(config_dict, port, method):
    ds_name = config_dict["data"].get("dataset", "mnist")
    ds_cfg = get_dataset_config(ds_name)
    nc = ds_cfg.num_classes
    n_clients = config_dict["server"]["min_clients"]
    n_rounds = config_dict["experiment"].get("num_rounds", 40)
    cid_to_model, cid_to_family, cid_to_rate, family_model_names = _parse_clients_from_config(config_dict)
    enable_fedgen = method in ("afad", "fedgen")
    enable_heterofl = method in ("afad", "heterofl")
    device = get_device()
    generator = FedGenGenerator(noise_dim=LATENT_DIM, num_classes=nc, latent_dim=LATENT_DIM) if enable_fedgen else None
    model_factories = build_wrapped_model_factories(family_model_names) if enable_fedgen else build_plain_model_factories(cid_to_model)
    first_name = next(iter(family_model_names.values()))
    init_model = FedGenModelWrapper(ModelRegistry.create_model(first_name, num_classes=nc), latent_dim=LATENT_DIM, num_classes=nc) if enable_fedgen else ModelRegistry.create_model(first_name, num_classes=nc)
    init_params = fl.common.ndarrays_to_parameters([v.cpu().numpy() for v in init_model.state_dict().values()])
    fg_cfg = config_dict.get("strategy", {}).get("fedgen", {})
    fedgen_config = {"gen_lr": fg_cfg.get("gen_lr", 3e-4), "batch_size": fg_cfg.get("batch_size", 128), "ensemble_alpha": fg_cfg.get("ensemble_alpha", 1.0), "ensemble_eta": fg_cfg.get("ensemble_eta", 1.0), "gen_epochs": fg_cfg.get("gen_epochs", 2), "teacher_iters": fg_cfg.get("teacher_iters", 25), "device": device}
    tr_cfg = config_dict.get("training", {})
    training_config = {"lr": tr_cfg.get("learning_rate", 0.01), "momentum": tr_cfg.get("momentum", 0.9), "weight_decay": tr_cfg.get("weight_decay", 0.0005), "local_epochs": tr_cfg.get("local_epochs", 3), "fedprox_mu": tr_cfg.get("fedprox_mu", 0.0)}
    strategy = AFADStrategy(initial_parameters=init_params, generator=generator, model_factories=model_factories, client_model_rates=cid_to_rate if enable_heterofl else None, family_model_names=family_model_names, fedgen_config=fedgen_config, training_config=training_config, enable_fedgen=enable_fedgen, enable_heterofl=enable_heterofl, num_rounds=n_rounds, num_classes=nc, min_fit_clients=n_clients, min_available_clients=n_clients, fraction_fit=1.0, fraction_evaluate=1.0, min_evaluate_clients=n_clients, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn)
    for cid, fam in cid_to_family.items():
        strategy.set_client_family(cid, fam)
    logger.info(f"Starting server on port {port}, method={method}, clients={n_clients}")
    fl.server.start_server(server_address=f"0.0.0.0:{port}", config=fl.server.ServerConfig(num_rounds=n_rounds), strategy=strategy)
    summary = strategy.metrics_collector.summary()
    if summary:
        logger.info("Summary: " + ", ".join(f"{k}={v:.4f}" for k, v in summary.items()))

def run_client(config_dict, server_address, client_id, method):
    ds_name = config_dict["data"].get("dataset", "mnist")
    ds_cfg = get_dataset_config(ds_name)
    nc = ds_cfg.num_classes
    n_clients = config_dict["server"]["min_clients"]
    bs = config_dict["data"].get("batch_size", 64)
    epochs = config_dict.get("training", {}).get("local_epochs", 3)
    seed = config_dict["experiment"].get("seed", 42)
    cid_to_model, cid_to_family, cid_to_rate, _ = _parse_clients_from_config(config_dict)
    enable_fedgen = method in ("afad", "fedgen")
    enable_heterofl = method in ("afad", "heterofl")
    mn = cid_to_model.get(client_id, "heterofl_resnet18")
    fam = cid_to_family.get(client_id, "cnn")
    rate = cid_to_rate.get(client_id, 1.0)
    device = get_device()
    torch.manual_seed(seed); np.random.seed(seed)
    if ds_name == "organamnist":
        from src.data.medmnist_loader import load_organamnist_data
        dc = config_dict["data"]
        tl, test = load_organamnist_data(num_clients=n_clients, batch_size=bs, alpha=dc.get("dirichlet_alpha", 0.5), distribution=dc.get("distribution", "non_iid"), seed=seed)
    else:
        from src.data.mnist_loader import load_mnist_data
        tl, test = load_mnist_data(num_clients=n_clients, batch_size=bs)
    train_loader = tl[int(client_id) % len(tl)]
    base = ModelRegistry.create_model(mn, num_classes=nc, model_rate=rate) if enable_heterofl else ModelRegistry.create_model(mn, num_classes=nc)
    if enable_fedgen:
        model = FedGenModelWrapper(base, latent_dim=LATENT_DIM, num_classes=nc)
        gen = FedGenGenerator(noise_dim=LATENT_DIM, num_classes=nc, latent_dim=LATENT_DIM)
    else:
        model, gen = base, None
    if enable_heterofl and enable_fedgen:
        kd = 0.5 / rate
        client = AFADClient(cid=client_id, model=model, generator=gen, train_loader=train_loader, val_loader=test, epochs=epochs, device=device, family=fam, model_rate=rate, model_name=mn, num_classes=nc, generative_alpha=kd, generative_beta=kd)
    elif enable_fedgen:
        client = FedGenClient(cid=client_id, model=model, generator=gen, train_loader=train_loader, val_loader=test, epochs=epochs, device=device, num_classes=nc, family=fam)
    else:
        client = HeteroFLClient(cid=client_id, model=model, train_loader=train_loader, val_loader=test, epochs=epochs, device=device, family=fam, model_rate=rate, model_name=mn, num_classes=nc)
    logger.info(f"Client {client_id}: {mn}, {fam}, rate={rate}, connecting to {server_address}")
    fl.client.start_client(server_address=server_address, client=client.to_client())

def main():
    p = argparse.ArgumentParser(description="AFAD Distributed FL")
    p.add_argument("role", choices=["server", "client"])
    p.add_argument("--config", default="config/afad_config.yaml")
    p.add_argument("--method", default="afad", choices=["afad", "heterofl", "fedgen"])
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--server-address", default="localhost:8080")
    p.add_argument("--client-id", default="0")
    a = p.parse_args()
    cfg = load_config(a.config)
    if a.role == "server": run_server(cfg, a.port, a.method)
    else: run_client(cfg, a.server_address, a.client_id, a.method)

if __name__ == "__main__":
    main()
