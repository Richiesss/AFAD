"""Direct FL simulation without Flower/Ray.

Bypasses start_simulation() to avoid Ray worker venv re-installation.
Manually runs FL rounds: distribute -> client fit -> aggregate -> generator train.

Usage:
    python scripts/run_direct_sim.py
"""

import math
import os
import pickle
import sys
import time

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

# ── Experiment config ────────────────────────────────────────
NUM_CLIENTS = 4
NUM_ROUNDS = 5
LOCAL_EPOCHS = 1
SEED = 42
LATENT_DIM = 32
NUM_CLASSES = 10
FEDGEN_WARMUP_ROUNDS = 3
LR_BASE = 0.01
MAX_SAMPLES = 500

CID_TO_MODEL = {
    "0": "heterofl_resnet18",
    "1": "heterofl_resnet18",
    "2": "heterofl_vit_small",
    "3": "heterofl_vit_small",
}
CID_TO_FAMILY = {"0": "cnn", "1": "cnn", "2": "vit", "3": "vit"}
CID_TO_RATE = {"0": 1.0, "1": 0.5, "2": 1.0, "3": 0.5}
FAMILY_MODEL_NAMES = {"cnn": "heterofl_resnet18", "vit": "heterofl_vit_small"}


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
) -> tuple:
    """Evaluate all family global models on test set."""
    total_correct = 0
    total_loss = 0.0
    total_samples = 0
    criterion = torch.nn.CrossEntropyLoss()

    for family, np_params in family_globals.items():
        model_name = FAMILY_MODEL_NAMES[family]
        base = ModelRegistry.create_model(model_name, num_classes=NUM_CLASSES)
        model = (
            FedGenModelWrapper(base, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES)
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
) -> list:
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
    for family, model_name in FAMILY_MODEL_NAMES.items():
        base = ModelRegistry.create_model(model_name, num_classes=NUM_CLASSES)
        model = (
            FedGenModelWrapper(base, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES)
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
            noise_dim=LATENT_DIM, num_classes=NUM_CLASSES, latent_dim=LATENT_DIM
        )
        gen_trainer = AFADGeneratorTrainer(
            generator=generator, gen_lr=3e-4, batch_size=128, device=device
        )

    client_label_counts: dict = {}
    results = []

    for round_num in range(1, NUM_ROUNDS + 1):
        t0 = time.time()
        lr = cosine_lr(LR_BASE, round_num, NUM_ROUNDS)

        # Serialize generator params for clients
        gen_params_bytes = None
        use_regularization = False
        if (
            enable_fedgen
            and generator_trained
            and round_num > FEDGEN_WARMUP_ROUNDS
            and generator is not None
        ):
            gen_params = [v.cpu().numpy() for v in generator.state_dict().values()]
            gen_params_bytes = pickle.dumps(gen_params)
            use_regularization = True

        # ── Client training ───────────────────────────────────
        client_updates: dict = {}

        for cid in sorted(CID_TO_MODEL.keys()):
            family = CID_TO_FAMILY[cid]
            model_name = CID_TO_MODEL[cid]
            model_rate = CID_TO_RATE[cid]

            if enable_heterofl:
                base = ModelRegistry.create_model(
                    model_name, num_classes=NUM_CLASSES, model_rate=model_rate
                )
            else:
                base = ModelRegistry.create_model(model_name, num_classes=NUM_CLASSES)

            model = (
                FedGenModelWrapper(base, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES)
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
                "weight_decay": 0.0001,
                "local_epochs": LOCAL_EPOCHS,
                "fedprox_mu": 0.0,
                "model_rate": model_rate,
                "regularization": use_regularization,
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
                    epochs=LOCAL_EPOCHS,
                    device=device,
                    family=family,
                    model_rate=model_rate,
                    model_name=model_name,
                    num_classes=NUM_CLASSES,
                )
            elif enable_fedgen:
                client = FedGenClient(
                    cid=cid,
                    model=model,
                    generator=generator,
                    train_loader=train_loader,
                    epochs=LOCAL_EPOCHS,
                    device=device,
                    num_classes=NUM_CLASSES,
                )
            else:
                client = HeteroFLClient(
                    cid=cid,
                    model=model,
                    train_loader=train_loader,
                    epochs=LOCAL_EPOCHS,
                    device=device,
                    family=family,
                    model_rate=model_rate,
                    model_name=model_name,
                    num_classes=NUM_CLASSES,
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
                model_name = FAMILY_MODEL_NAMES[family]
                base = ModelRegistry.create_model(model_name, num_classes=NUM_CLASSES)
                wrapped = FedGenModelWrapper(
                    base, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES
                )
                sd = wrapped.state_dict()
                for k, p in zip(list(sd.keys()), np_params):
                    sd[k] = torch.from_numpy(p.copy())
                wrapped.load_state_dict(sd)
                wrapped.to(device)
                torch_models[family] = wrapped

            family_lc: dict = {}
            for cid, counts in client_label_counts.items():
                fam = CID_TO_FAMILY.get(cid)
                if fam and fam in torch_models:
                    if fam not in family_lc:
                        family_lc[fam] = [0] * NUM_CLASSES
                    for i, c in enumerate(counts):
                        if i < NUM_CLASSES:
                            family_lc[fam][i] += c

            lc_list = [family_lc[f] for f in torch_models if f in family_lc]
            if lc_list:
                label_weights, qualified = gen_trainer.get_label_weights(
                    lc_list, num_classes=NUM_CLASSES
                )
            else:
                nm = len(torch_models)
                label_weights = np.ones((NUM_CLASSES, nm)) / nm
                qualified = list(range(NUM_CLASSES))

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
            family_globals, test_loader, device, enable_fedgen
        )
        elapsed = time.time() - t0
        results.append(
            {"round": round_num, "accuracy": acc, "loss": loss, "wall_time": elapsed}
        )
        print(
            f"  Round {round_num}/{NUM_ROUNDS}: "
            f"acc={acc:.4f}  loss={loss:.4f}  time={elapsed:.1f}s",
            flush=True,
        )

    return results


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


def main() -> None:
    print(f"Device: {get_device()}", flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(
        f"Loading MNIST ({MAX_SAMPLES} samples, {NUM_CLIENTS} clients)...", flush=True
    )
    train_loaders, test_loader = load_mnist_data(
        num_clients=NUM_CLIENTS, batch_size=32, max_samples=MAX_SAMPLES
    )
    print(f"Test set size: {len(test_loader.dataset)}", flush=True)

    all_results: dict = {}

    for label, fedgen, heterofl in [
        ("HeteroFL Only", False, True),
        ("FedGen Only", True, False),
        ("AFAD Hybrid", True, True),
    ]:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        all_results[label] = run_experiment(
            label=label,
            train_loaders=train_loaders,
            test_loader=test_loader,
            enable_fedgen=fedgen,
            enable_heterofl=heterofl,
        )

    print_table(all_results)


if __name__ == "__main__":
    main()
