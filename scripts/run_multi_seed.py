"""
Multi-seed statistical validation for AFAD Phase 2.

Runs Phase 2 experiments across multiple seeds and performs
statistical analysis (mean/std, 95% CI, paired t-test) to
validate that AFAD Hybrid significantly outperforms baselines.

Results are saved incrementally to JSON for crash recovery.

Usage:
    python scripts/run_multi_seed.py config/afad_phase2_config.yaml
    python scripts/run_multi_seed.py config/afad_phase2_config.yaml --seeds 42 123 456
    python scripts/run_multi_seed.py config/afad_phase2_config.yaml --output results/my_results.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_comparison import (
    get_client_mappings,
    print_comparison_table,
    run_single_experiment,
)
from src.data.dataset_config import get_dataset_config
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger("MultiSeed")

DEFAULT_SEEDS = [42, 123, 456, 789, 1024, 2025, 3141, 4096, 5555, 7777]

METHODS = [
    {"label": "HeteroFL Only", "enable_fedgen": False, "enable_heterofl": True},
    {"label": "KD Only", "enable_fedgen": True, "enable_heterofl": False},
    {"label": "AFAD Hybrid", "enable_fedgen": True, "enable_heterofl": True},
]


def load_existing_results(output_path: Path) -> dict:
    """Load previously saved results for crash recovery."""
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)
        logger.info(
            f"Loaded existing results: {len(data.get('seeds_completed', []))} seeds completed"
        )
        return data
    return {"seeds_completed": [], "seed_results": {}}


def save_results(output_path: Path, results: dict) -> None:
    """Save results incrementally to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")


def extract_metrics(rounds: list[dict]) -> dict:
    """Extract best and final accuracy/loss from per-round metrics."""
    best_acc = max(r["accuracy"] for r in rounds)
    final_acc = rounds[-1]["accuracy"]
    best_loss = min(r["loss"] for r in rounds)
    final_loss = rounds[-1]["loss"]
    total_time = sum(r["wall_time"] for r in rounds)
    return {
        "best_accuracy": best_acc,
        "final_accuracy": final_acc,
        "best_loss": best_loss,
        "final_loss": final_loss,
        "total_time": total_time,
    }


def run_seed(
    seed: int,
    config_dict: dict,
) -> dict[str, list[dict]]:
    """Run all 3 methods for a single seed and return per-method rounds."""
    dataset_name = config_dict["data"].get("dataset", "mnist")
    ds_cfg = get_dataset_config(dataset_name)
    num_classes = ds_cfg.num_classes
    num_clients = config_dict["server"]["min_clients"]
    num_rounds = config_dict["experiment"].get("num_rounds", 40)
    local_epochs = config_dict.get("training", {}).get("local_epochs", 3)
    batch_size = config_dict["data"].get("batch_size", 64)
    fedprox_mu = config_dict.get("training", {}).get("fedprox_mu", 0.0)

    cid_to_model, cid_to_family = get_client_mappings(num_clients)

    # Load data with this seed
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
        from src.data.mnist_loader import load_mnist_data

        train_loaders, test_loader = load_mnist_data(
            num_clients=num_clients,
            batch_size=batch_size,
        )

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

    seed_results = {}
    for method in METHODS:
        rounds = run_single_experiment(
            label=f"{method['label']} (seed={seed})",
            enable_fedgen=method["enable_fedgen"],
            enable_heterofl=method.get("enable_heterofl", True),
            **exp_kwargs,
        )
        seed_results[method["label"]] = rounds

    return seed_results


def compute_statistics(all_results: dict) -> dict:
    """Compute mean, std, CI, and paired t-tests from all seed results."""
    method_names = [m["label"] for m in METHODS]
    method_metrics: dict[str, dict[str, list[float]]] = {
        name: {"best_accuracy": [], "final_accuracy": []} for name in method_names
    }

    for seed_data in all_results["seed_results"].values():
        for method_name in method_names:
            if method_name not in seed_data:
                continue
            metrics = extract_metrics(seed_data[method_name])
            method_metrics[method_name]["best_accuracy"].append(
                metrics["best_accuracy"]
            )
            method_metrics[method_name]["final_accuracy"].append(
                metrics["final_accuracy"]
            )

    summary: dict = {"methods": {}, "tests": []}

    for name in method_names:
        best_accs = np.array(method_metrics[name]["best_accuracy"])
        final_accs = np.array(method_metrics[name]["final_accuracy"])
        n = len(best_accs)

        if n < 2:
            summary["methods"][name] = {
                "n": n,
                "best_mean": float(best_accs.mean()) if n > 0 else 0.0,
                "best_std": 0.0,
                "final_mean": float(final_accs.mean()) if n > 0 else 0.0,
                "final_std": 0.0,
            }
            continue

        t_crit = stats.t.ppf(0.975, df=n - 1)
        best_ci = t_crit * best_accs.std(ddof=1) / np.sqrt(n)
        final_ci = t_crit * final_accs.std(ddof=1) / np.sqrt(n)

        summary["methods"][name] = {
            "n": n,
            "best_mean": float(best_accs.mean()),
            "best_std": float(best_accs.std(ddof=1)),
            "best_ci_95": float(best_ci),
            "final_mean": float(final_accs.mean()),
            "final_std": float(final_accs.std(ddof=1)),
            "final_ci_95": float(final_ci),
        }

    # Paired t-tests: AFAD vs each baseline
    afad_best = np.array(method_metrics["AFAD Hybrid"]["best_accuracy"])
    afad_final = np.array(method_metrics["AFAD Hybrid"]["final_accuracy"])

    for baseline in ["HeteroFL Only", "KD Only"]:
        base_best = np.array(method_metrics[baseline]["best_accuracy"])
        base_final = np.array(method_metrics[baseline]["final_accuracy"])

        if len(afad_best) < 2 or len(base_best) < 2:
            continue
        if len(afad_best) != len(base_best):
            continue

        t_best, p_best = stats.ttest_rel(afad_best, base_best)
        t_final, p_final = stats.ttest_rel(afad_final, base_final)

        summary["tests"].append(
            {
                "comparison": f"AFAD vs {baseline}",
                "best_accuracy": {
                    "t_statistic": float(t_best),
                    "p_value": float(p_best),
                    "significant": bool(p_best < 0.05),
                },
                "final_accuracy": {
                    "t_statistic": float(t_final),
                    "p_value": float(p_final),
                    "significant": bool(p_final < 0.05),
                },
            }
        )

    return summary


def print_statistical_summary(summary: dict, num_seeds: int) -> None:
    """Print formatted statistical summary to stdout."""
    print(f"\n{'=' * 70}")
    print(f"  Statistical Summary ({num_seeds} seeds)")
    print(f"{'=' * 70}")

    header = f"{'Method':<20} {'Best Acc (Mean±Std)':>22} {'Final Acc (Mean±Std)':>22}"
    print(header)
    print("-" * len(header))

    for name, m in summary["methods"].items():
        n = m["n"]
        if n < 2:
            best_str = f"{m['best_mean'] * 100:.2f}% (n={n})"
            final_str = f"{m['final_mean'] * 100:.2f}% (n={n})"
        else:
            best_str = f"{m['best_mean'] * 100:.2f} ± {m['best_std'] * 100:.2f}%"
            final_str = f"{m['final_mean'] * 100:.2f} ± {m['final_std'] * 100:.2f}%"
        print(f"{name:<20} {best_str:>22} {final_str:>22}")

    if summary["tests"]:
        print(f"\n{'=' * 70}")
        print("  Paired t-test (AFAD vs baselines)")
        print(f"{'=' * 70}")

        for test in summary["tests"]:
            comp = test["comparison"]
            for metric_key, metric_label in [
                ("best_accuracy", "Best Acc"),
                ("final_accuracy", "Final Acc"),
            ]:
                t_val = test[metric_key]["t_statistic"]
                p_val = test[metric_key]["p_value"]
                sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
                print(f"  {comp} ({metric_label}): t={t_val:.3f}, p={p_val:.4f} {sig}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-seed statistical validation for AFAD"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="List of random seeds (default: 10 seeds)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/multi_seed_results.json",
        help="Output JSON path (default: results/multi_seed_results.json)",
    )
    args = parser.parse_args()

    config_dict = load_config(args.config)
    output_path = Path(args.output)
    seeds = args.seeds

    logger.info(f"Config: {args.config}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Output: {output_path}")

    # Load existing results for resume
    all_results = load_existing_results(output_path)
    completed_seeds = set(all_results["seeds_completed"])

    total_seeds = len(seeds)
    remaining = [s for s in seeds if s not in completed_seeds]

    if completed_seeds:
        logger.info(
            f"Resuming: {len(completed_seeds)}/{total_seeds} seeds already completed"
        )

    if not remaining:
        logger.info("All seeds already completed. Computing statistics.")
    else:
        logger.info(f"Running {len(remaining)} remaining seeds: {remaining}")

    start_time = time.time()

    for i, seed in enumerate(remaining):
        logger.info(f"\n{'#' * 60}")
        logger.info(f"Seed {seed} ({len(completed_seeds) + i + 1}/{total_seeds})")
        logger.info(f"{'#' * 60}")

        seed_start = time.time()
        seed_results = run_seed(seed, config_dict)
        seed_elapsed = time.time() - seed_start

        logger.info(f"Seed {seed} completed in {seed_elapsed:.1f}s")

        # Print per-seed comparison
        print_comparison_table(seed_results)

        # Save incrementally
        all_results["seed_results"][str(seed)] = seed_results
        all_results["seeds_completed"].append(seed)
        save_results(output_path, all_results)

    total_elapsed = time.time() - start_time
    logger.info(f"\nAll seeds completed in {total_elapsed:.1f}s")

    # Compute and display statistics
    summary = compute_statistics(all_results)
    all_results["statistical_summary"] = summary
    save_results(output_path, all_results)

    print_statistical_summary(summary, len(all_results["seeds_completed"]))


if __name__ == "__main__":
    main()
