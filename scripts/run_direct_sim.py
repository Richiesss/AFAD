"""Direct FL simulation without Flower/Ray.

Bypasses start_simulation() to avoid Ray worker venv re-installation.
Manually runs FL rounds: distribute -> client fit -> aggregate -> generator train.

Usage:
    python scripts/run_direct_sim.py            # full 10-client 30-round experiment
    python scripts/run_direct_sim.py --quick    # 4 clients, 5 rounds (fast validation)
    python scripts/run_direct_sim.py --output results/my_run.json
"""

import json
import math
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from flwr.common import parameters_to_ndarrays

import src.models.cnn.heterofl_resnet  # noqa: F401
import src.models.vit.heterofl_vit  # noqa: F401
from src.client.afad_client import AFADClient
from src.client.fedgen_client import FedGenClient
from src.client.heterofl_client import HeteroFLClient
from src.data.mnist_loader import load_mnist_data
from src.models.fedgen_wrapper import FedGenModelWrapper
from src.models.registry import ModelRegistry
from src.server.generator.afad_generator_trainer import AFADGeneratorTrainer
from src.server.generator.fedgen_generator import FedGenGenerator
from src.server.strategy.heterofl_aggregator import HeteroFLAggregator

# ── Full experiment config ────────────────────────────────────
# 10 clients (5 CNN + 5 ViT), rates [1.0, 1.0, 0.5, 0.5, 0.25] per family.
# 2 full-rate clients per family give the bottleneck enough training signal
# for AFAD's shared latent space, while sub-rate clients test HeteroFL efficiency.
NUM_CLIENTS = 10
NUM_ROUNDS = 30
LOCAL_EPOCHS = 1
SEED = 42
LATENT_DIM = 32
NUM_CLASSES = 10
FEDGEN_WARMUP_ROUNDS = 1
AFAD_KD_WARMUP_ROUNDS = 5  # Delay AFAD KD until generator has converged
LR_BASE = 0.01
MAX_SAMPLES = 5000

#  cid  family  rate
#   0    cnn    1.00
#   1    cnn    1.00
#   2    cnn    0.50
#   3    cnn    0.50
#   4    cnn    0.25
#   5    vit    1.00
#   6    vit    1.00
#   7    vit    0.50
#   8    vit    0.50
#   9    vit    0.25
CID_TO_MODEL = {
    "0": "heterofl_resnet18", "1": "heterofl_resnet18",
    "2": "heterofl_resnet18", "3": "heterofl_resnet18",
    "4": "heterofl_resnet18",
    "5": "heterofl_vit_small", "6": "heterofl_vit_small",
    "7": "heterofl_vit_small", "8": "heterofl_vit_small",
    "9": "heterofl_vit_small",
}
CID_TO_FAMILY = {
    "0": "cnn", "1": "cnn", "2": "cnn", "3": "cnn", "4": "cnn",
    "5": "vit", "6": "vit", "7": "vit", "8": "vit", "9": "vit",
}
CID_TO_RATE = {
    "0": 1.0, "1": 1.0, "2": 0.5, "3": 0.5, "4": 0.25,
    "5": 1.0, "6": 1.0, "7": 0.5, "8": 0.5, "9": 0.25,
}
FAMILY_MODEL_NAMES = {"cnn": "heterofl_resnet18", "vit": "heterofl_vit_small"}

# ── Quick-test config (4 clients, 5 rounds, 500 samples) ─────
_QUICK_NUM_CLIENTS = 4
_QUICK_MAX_SAMPLES = 500
_QUICK_CID_TO_MODEL = {
    "0": "heterofl_resnet18", "1": "heterofl_resnet18",
    "2": "heterofl_vit_small", "3": "heterofl_vit_small",
}
_QUICK_CID_TO_FAMILY = {"0": "cnn", "1": "cnn", "2": "vit", "3": "vit"}
_QUICK_CID_TO_RATE = {"0": 1.0, "1": 0.5, "2": 1.0, "3": 0.5}


def _default_cfg() -> dict:
    """Return default experiment configuration from module constants."""
    return {
        "num_rounds": NUM_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "num_classes": NUM_CLASSES,
        "latent_dim": LATENT_DIM,
        "fedgen_warmup_rounds": FEDGEN_WARMUP_ROUNDS,
        "afad_kd_warmup_rounds": AFAD_KD_WARMUP_ROUNDS,
        "lr_base": LR_BASE,
        "cid_to_model": dict(CID_TO_MODEL),
        "cid_to_family": dict(CID_TO_FAMILY),
        "cid_to_rate": dict(CID_TO_RATE),
        "family_model_names": dict(FAMILY_MODEL_NAMES),
    }


def _quick_cfg() -> dict:
    """Return quick-test configuration (4 clients, 5 rounds)."""
    return {
        "num_rounds": 5,
        "local_epochs": LOCAL_EPOCHS,
        "num_classes": NUM_CLASSES,
        "latent_dim": LATENT_DIM,
        "fedgen_warmup_rounds": 1,
        "lr_base": LR_BASE,
        "cid_to_model": dict(_QUICK_CID_TO_MODEL),
        "cid_to_family": dict(_QUICK_CID_TO_FAMILY),
        "cid_to_rate": dict(_QUICK_CID_TO_RATE),
        "family_model_names": dict(FAMILY_MODEL_NAMES),
    }


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def cosine_lr(base_lr: float, round_num: int, total_rounds: int) -> float:
    lr_min = base_lr * 0.01
    progress = round_num / total_rounds
    return lr_min + 0.5 * (base_lr - lr_min) * (1 + math.cos(math.pi * progress))


def evaluate_families(
    family_globals: dict,
    test_loader,
    device: str,
    enable_fedgen: bool,
    family_model_names: dict | None = None,
    num_classes: int = NUM_CLASSES,
    latent_dim: int = LATENT_DIM,
) -> tuple:
    """Evaluate all family global models on test set."""
    if family_model_names is None:
        family_model_names = FAMILY_MODEL_NAMES

    total_correct = 0
    total_loss = 0.0
    total_samples = 0
    criterion = torch.nn.CrossEntropyLoss()

    for family, np_params in family_globals.items():
        model_name = family_model_names[family]
        base = ModelRegistry.create_model(model_name, num_classes=num_classes)
        model = (
            FedGenModelWrapper(base, latent_dim=latent_dim, num_classes=num_classes)
            if enable_fedgen
            else base
        )
        sd = model.state_dict()
        for k, p in zip(list(sd.keys()), np_params):
            sd[k] = torch.from_numpy(p.copy())
        model.load_state_dict(sd)
        model.to(device)
        model.eval()

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                total_loss += criterion(out, labels).item() * labels.size(0)
                total_correct += out.max(1)[1].eq(labels).sum().item()
                total_samples += labels.size(0)

    acc = total_correct / total_samples if total_samples > 0 else 0.0
    loss = total_loss / total_samples if total_samples > 0 else 0.0
    return acc, loss


def run_experiment(
    label: str,
    train_loaders: list,
    test_loader,
    enable_fedgen: bool,
    enable_heterofl: bool,
    cfg: dict | None = None,
) -> tuple:
    """Run one FL experiment and return (per-round metrics, final family globals).

    Args:
        label: Experiment name for display.
        train_loaders: Per-client training data loaders.
        test_loader: Shared test data loader.
        enable_fedgen: Enable FedGen client-side KD and generator training.
        enable_heterofl: Enable HeteroFL width-scaled sub-models.
        cfg: Optional configuration dict to override defaults. Keys:
            num_rounds, local_epochs, num_classes, latent_dim,
            fedgen_warmup_rounds, lr_base, cid_to_model, cid_to_family,
            cid_to_rate, family_model_names.

    Returns:
        Tuple of (list of per-round metric dicts, dict of final family params).
    """
    # Resolve configuration (override defaults with any provided values)
    resolved = {**_default_cfg(), **(cfg or {})}
    num_rounds = resolved["num_rounds"]
    local_epochs = resolved["local_epochs"]
    num_classes = resolved["num_classes"]
    latent_dim = resolved["latent_dim"]
    fedgen_warmup_rounds = resolved["fedgen_warmup_rounds"]
    afad_kd_warmup_rounds = resolved.get("afad_kd_warmup_rounds", AFAD_KD_WARMUP_ROUNDS)
    lr_base = resolved["lr_base"]
    cid_to_model: dict = resolved["cid_to_model"]
    cid_to_family: dict = resolved["cid_to_family"]
    cid_to_rate: dict = resolved["cid_to_rate"]
    family_model_names: dict = resolved["family_model_names"]

    print(f"\n{'=' * 60}", flush=True)
    print(f"  {label}", flush=True)
    print(
        f"  enable_fedgen={enable_fedgen}, enable_heterofl={enable_heterofl}",
        flush=True,
    )
    print(f"{'=' * 60}", flush=True)

    device = get_device()
    aggregator = HeteroFLAggregator()
    num_preserved_tail = 2 if enable_fedgen else 1

    # Initialize family global models at rate=1.0
    family_globals: dict = {}
    for family, model_name in family_model_names.items():
        base = ModelRegistry.create_model(model_name, num_classes=num_classes)
        model = (
            FedGenModelWrapper(base, latent_dim=latent_dim, num_classes=num_classes)
            if enable_fedgen
            else base
        )
        family_globals[family] = [v.cpu().numpy() for v in model.state_dict().values()]

    # Initialize generator and trainer
    generator = None
    gen_trainer = None
    generator_trained = False
    if enable_fedgen:
        generator = FedGenGenerator(
            noise_dim=latent_dim, num_classes=num_classes, latent_dim=latent_dim
        )
        gen_trainer = AFADGeneratorTrainer(
            generator=generator, gen_lr=3e-4, batch_size=128, device=device
        )

    client_label_counts: dict = {}
    results = []

    for round_num in range(1, num_rounds + 1):
        t0 = time.time()
        lr = cosine_lr(lr_base, round_num, num_rounds)

        # Serialize generator params (EMA-smoothed) for clients
        gen_params_bytes = None
        use_regularization = False
        if (
            enable_fedgen
            and generator_trained
            and round_num > fedgen_warmup_rounds
            and generator is not None
            and gen_trainer is not None
        ):
            gen_params = [
                v.cpu().numpy() for v in gen_trainer.get_inference_state_dict().values()
            ]
            gen_params_bytes = pickle.dumps(gen_params)
            use_regularization = True

        # ── Client training ───────────────────────────────────
        client_updates: dict = {}

        for cid in sorted(cid_to_model.keys()):
            family = cid_to_family[cid]
            model_name = cid_to_model[cid]
            model_rate = cid_to_rate[cid]

            if enable_heterofl:
                base = ModelRegistry.create_model(
                    model_name, num_classes=num_classes, model_rate=model_rate
                )
            else:
                base = ModelRegistry.create_model(model_name, num_classes=num_classes)

            model = (
                FedGenModelWrapper(base, latent_dim=latent_dim, num_classes=num_classes)
                if enable_fedgen
                else base
            )

            # Distribute from family global
            if enable_heterofl and family in family_globals:
                sub_params = aggregator.distribute(
                    family_globals[family],
                    cid,
                    model_rate,
                    num_preserved_tail_layers=num_preserved_tail,
                )
            else:
                sub_params = family_globals.get(family, [])

            config = {
                "round": round_num,
                "family": family,
                "lr": lr,
                "momentum": 0.9,
                "weight_decay": 0.0005,
                "local_epochs": local_epochs,
                "fedprox_mu": 0.0,
                "model_rate": model_rate,
                "regularization": (
                    use_regularization and (round_num > afad_kd_warmup_rounds)
                    if (enable_heterofl and enable_fedgen)
                    else use_regularization
                ),
                "use_local_init": (round_num == 1 and len(sub_params) == 0),
            }
            if gen_params_bytes is not None:
                config["generator_params"] = gen_params_bytes

            train_loader = train_loaders[int(cid) % len(train_loaders)]

            if enable_heterofl and enable_fedgen:
                client = AFADClient(
                    cid=cid,
                    model=model,
                    generator=generator,
                    train_loader=train_loader,
                    epochs=local_epochs,
                    device=device,
                    family=family,
                    model_rate=model_rate,
                    model_name=model_name,
                    num_classes=num_classes,
                    generative_alpha=1.0,
                    generative_beta=1.0,
                )
            elif enable_fedgen:
                client = FedGenClient(
                    cid=cid,
                    model=model,
                    generator=generator,
                    train_loader=train_loader,
                    epochs=local_epochs,
                    device=device,
                    num_classes=num_classes,
                )
            else:
                client = HeteroFLClient(
                    cid=cid,
                    model=model,
                    train_loader=train_loader,
                    epochs=local_epochs,
                    device=device,
                    family=family,
                    model_rate=model_rate,
                    model_name=model_name,
                    num_classes=num_classes,
                )

            params, num_ex, metrics = client.fit(sub_params, config)

            lc_str = metrics.get("label_counts", "")
            if isinstance(lc_str, str) and lc_str:
                client_label_counts[cid] = [int(x) for x in lc_str.split(",") if x]

            if family not in client_updates:
                client_updates[family] = []
            client_updates[family].append((cid, params, num_ex))

        # ── Aggregate ─────────────────────────────────────────
        if enable_heterofl:
            for family, items in client_updates.items():
                if family not in family_globals:
                    continue
                label_splits = {}
                for cid, _, _ in items:
                    counts = client_label_counts.get(cid)
                    if counts:
                        label_splits[cid] = [i for i, c in enumerate(counts) if c > 0]
                agg = aggregator.aggregate(
                    family=family,
                    results=items,
                    global_params=family_globals[family],
                    client_label_splits=label_splits or None,
                    num_preserved_tail_layers=num_preserved_tail,
                )
                family_globals[family] = parameters_to_ndarrays(agg)
        else:
            for family, items in client_updates.items():
                total_ex = sum(n for _, _, n in items)
                if total_ex == 0:
                    continue
                avg = [np.zeros_like(p, dtype=np.float64) for p in items[0][1]]
                for _, params, n in items:
                    w = n / total_ex
                    for i, p in enumerate(params):
                        avg[i] += w * p
                family_globals[family] = avg

        # ── Generator training ────────────────────────────────
        if enable_fedgen and gen_trainer is not None and len(family_globals) >= 2:
            torch_models = {}
            for family, np_params in family_globals.items():
                model_name = family_model_names[family]
                base = ModelRegistry.create_model(model_name, num_classes=num_classes)
                wrapped = FedGenModelWrapper(
                    base, latent_dim=latent_dim, num_classes=num_classes
                )
                sd = wrapped.state_dict()
                for k, p in zip(list(sd.keys()), np_params):
                    sd[k] = torch.from_numpy(p.copy())
                wrapped.load_state_dict(sd)
                wrapped.to(device)
                torch_models[family] = wrapped

            family_lc: dict = {}
            for cid, counts in client_label_counts.items():
                fam = cid_to_family.get(cid)
                if fam and fam in torch_models:
                    if fam not in family_lc:
                        family_lc[fam] = [0] * num_classes
                    for i, c in enumerate(counts):
                        if i < num_classes:
                            family_lc[fam][i] += c

            lc_list = [family_lc[f] for f in torch_models if f in family_lc]
            if lc_list:
                label_weights, qualified = gen_trainer.get_label_weights(
                    lc_list, num_classes=num_classes
                )
            else:
                nm = len(torch_models)
                label_weights = np.ones((num_classes, nm)) / nm
                qualified = list(range(num_classes))

            gen_trainer.train_generator(
                models=torch_models,
                label_weights=label_weights,
                qualified_labels=qualified,
                num_epochs=1,
                num_teacher_iters=10,
            )
            generator_trained = True

        # ── Evaluate ──────────────────────────────────────────
        acc, loss = evaluate_families(
            family_globals,
            test_loader,
            device,
            enable_fedgen,
            family_model_names=family_model_names,
            num_classes=num_classes,
            latent_dim=latent_dim,
        )
        elapsed = time.time() - t0
        results.append(
            {"round": round_num, "accuracy": acc, "loss": loss, "wall_time": elapsed}
        )
        print(
            f"  Round {round_num}/{num_rounds}: "
            f"acc={acc:.4f}  loss={loss:.4f}  time={elapsed:.1f}s",
            flush=True,
        )

    return results, family_globals


def print_table(all_results: dict) -> None:
    labels = list(all_results.keys())
    num_rounds = max(len(v) for v in all_results.values())
    header = f"{'Round':>5}" + "".join(f" | {lb:>20}" for lb in labels)
    sep = "-" * len(header)
    print(f"\n{'=' * len(header)}")
    print("  COMPARISON: Accuracy per Round")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)
    for i in range(num_rounds):
        row = f"{i + 1:>5}"
        for lb in labels:
            acc = all_results[lb][i]["accuracy"] if i < len(all_results[lb]) else None
            row += f" | {acc:>19.2%}" if acc is not None else f" | {'N/A':>19}"
        print(row)
    print(sep)
    print(
        f"{'BEST':>5}"
        + "".join(
            f" | {max(r['accuracy'] for r in all_results[lb]):>19.2%}" for lb in labels
        )
    )
    print(
        f"{'FINAL':>5}"
        + "".join(f" | {all_results[lb][-1]['accuracy']:>19.2%}" for lb in labels)
    )
    print()
    for lb in labels:
        total_t = sum(r["wall_time"] for r in all_results[lb])
        print(f"  {lb}: total={total_t:.0f}s")


def save_results(all_results: dict, output_path: Path) -> None:
    """Persist per-round experiment results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {output_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Direct FL simulation (no Ray)")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test: 4 clients, 5 rounds, 500 samples",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: results/direct_sim[_quick]_<timestamp>.json)",
    )
    args = parser.parse_args()

    print(f"Device: {get_device()}", flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if args.quick:
        num_clients = _QUICK_NUM_CLIENTS
        max_samples = _QUICK_MAX_SAMPLES
        exp_cfg = _quick_cfg()
        print("Quick test mode: 4 clients, 5 rounds, 500 samples", flush=True)
    else:
        num_clients = NUM_CLIENTS
        max_samples = MAX_SAMPLES
        exp_cfg = None  # use defaults

    print(
        f"Loading MNIST ({max_samples} samples, {num_clients} clients)...", flush=True
    )
    train_loaders, test_loader = load_mnist_data(
        num_clients=num_clients, batch_size=32, max_samples=max_samples
    )
    print(f"Test set size: {len(test_loader.dataset)}", flush=True)

    all_results: dict = {}
    all_final_globals: dict = {}

    for label, fedgen, heterofl in [
        ("HeteroFL Only", False, True),
        ("FedGen Only", True, False),
        ("AFAD Hybrid", True, True),
    ]:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        rounds, final_globals = run_experiment(
            label=label,
            train_loaders=train_loaders,
            test_loader=test_loader,
            enable_fedgen=fedgen,
            enable_heterofl=heterofl,
            cfg=exp_cfg,
        )
        all_results[label] = rounds
        all_final_globals[label] = final_globals

    print_table(all_results)

    # Show model divergence between methods
    print("\n  Model parameter divergence (FedGen vs AFAD, final round):")
    fg_g = all_final_globals.get("FedGen Only", {})
    afad_g = all_final_globals.get("AFAD Hybrid", {})
    for fam in FAMILY_MODEL_NAMES:
        if fam in fg_g and fam in afad_g:
            diffs = [np.linalg.norm(a - b) for a, b in zip(fg_g[fam], afad_g[fam])]
            total_diff = sum(diffs)
            total_norm = sum(np.linalg.norm(p) for p in fg_g[fam])
            print(
                f"    {fam}: L2_diff={total_diff:.2f}, "
                f"relative={total_diff / total_norm:.3f}"
            )

    # Persist results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_quick" if args.quick else ""
    output_path = (
        Path(args.output)
        if args.output
        else Path(f"results/direct_sim{suffix}_{timestamp}.json")
    )
    save_results(all_results, output_path)


if __name__ == "__main__":
    main()
