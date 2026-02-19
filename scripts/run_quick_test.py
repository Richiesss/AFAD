"""Quick minimal experiment: 4 clients, 5 rounds, MNIST.

Tests all 3 modes (HeteroFL Only, FedGen Only, AFAD Hybrid) with
the smallest possible configuration for fast validation.

Uses direct simulation (no Ray/Flower) for fast startup.

Usage:
    python scripts/run_quick_test.py
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

import numpy as np
import torch

from scripts.run_direct_sim import (
    _QUICK_MAX_SAMPLES,
    _QUICK_NUM_CLIENTS,
    SEED,
    _quick_cfg,
    get_device,
    print_table,
    run_experiment,
    save_results,
)
from src.data.mnist_loader import load_mnist_data


def main() -> None:
    print(f"Device: {get_device()}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(
        f"Loading MNIST for {_QUICK_NUM_CLIENTS} clients "
        f"(max {_QUICK_MAX_SAMPLES} samples)..."
    )
    train_loaders, test_loader = load_mnist_data(
        num_clients=_QUICK_NUM_CLIENTS,
        batch_size=32,
        max_samples=_QUICK_MAX_SAMPLES,
    )

    exp_cfg = _quick_cfg()
    all_results: dict = {}

    for label, fedgen, heterofl in [
        ("HeteroFL Only", False, True),
        ("FedGen Only", True, False),
        ("AFAD Hybrid", True, True),
    ]:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        rounds, _ = run_experiment(
            label=label,
            train_loaders=train_loaders,
            test_loader=test_loader,
            enable_fedgen=fedgen,
            enable_heterofl=heterofl,
            cfg=exp_cfg,
        )
        all_results[label] = rounds

    print_table(all_results)
    save_results(all_results, Path("results/quick_test.json"))


if __name__ == "__main__":
    main()
