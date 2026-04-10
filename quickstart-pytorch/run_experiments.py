"""Batch experiment runner for federated learning simulations.

Runs a predefined set of experiments with different combinations of
strategies, datasets, models, partitioners, and client selection algorithms.
Each experiment produces a metrics_<name>.json file. After all experiments
finish, a comparison summary is printed.

Usage:
    python run_experiments.py                 # run all experiments
    python run_experiments.py --dry-run       # print commands without running
    python run_experiments.py --filter fedprox # only run experiments containing "fedprox"
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ======================== Experiment Definitions ========================

EXPERIMENTS = [
    # --- Baseline comparisons: Strategy × IID ---
    {
        "name": "fedavg_cifar10_iid_random",
        "config": {
            "strategy": "fedavg",
            "dataset": "cifar10",
            "model": "cnn",
            "partitioner": "iid",
            "client-selection": "random",
            "num-server-rounds": 20,
        },
    },
    {
        "name": "fedprox_cifar10_iid_random",
        "config": {
            "strategy": "fedprox",
            "dataset": "cifar10",
            "model": "cnn",
            "partitioner": "iid",
            "client-selection": "random",
            "proximal-mu": 0.1,
            "num-server-rounds": 20,
        },
    },
    {
        "name": "fedadagrad_cifar10_iid_random",
        "config": {
            "strategy": "fedadagrad",
            "dataset": "cifar10",
            "model": "cnn",
            "partitioner": "iid",
            "client-selection": "random",
            "num-server-rounds": 20,
        },
    },

    # --- Non-IID comparisons: Strategy × Dirichlet ---
    {
        "name": "fedavg_cifar10_noniid_random",
        "config": {
            "strategy": "fedavg",
            "dataset": "cifar10",
            "model": "cnn",
            "partitioner": "dirichlet",
            "dirichlet-alpha": 0.3,
            "client-selection": "random",
            "num-server-rounds": 20,
        },
    },
    {
        "name": "fedprox_cifar10_noniid_random",
        "config": {
            "strategy": "fedprox",
            "dataset": "cifar10",
            "model": "cnn",
            "partitioner": "dirichlet",
            "dirichlet-alpha": 0.3,
            "client-selection": "random",
            "proximal-mu": 0.1,
            "num-server-rounds": 20,
        },
    },

    # --- Client selection comparisons under non-IID ---
    {
        "name": "fedavg_cifar10_noniid_highloss",
        "config": {
            "strategy": "fedavg",
            "dataset": "cifar10",
            "model": "cnn",
            "partitioner": "dirichlet",
            "dirichlet-alpha": 0.3,
            "client-selection": "high-loss",
            "num-server-rounds": 20,
        },
    },
    {
        "name": "fedavg_cifar10_noniid_cluster",
        "config": {
            "strategy": "fedavg",
            "dataset": "cifar10",
            "model": "cnn",
            "partitioner": "dirichlet",
            "dirichlet-alpha": 0.3,
            "client-selection": "cluster-based",
            "num-server-rounds": 20,
        },
    },
    {
        "name": "fedavg_cifar10_noniid_poc",
        "config": {
            "strategy": "fedavg",
            "dataset": "cifar10",
            "model": "cnn",
            "partitioner": "dirichlet",
            "dirichlet-alpha": 0.3,
            "client-selection": "power-of-choice",
            "num-server-rounds": 20,
        },
    },

    # --- Model comparison ---
    {
        "name": "fedavg_cifar10_iid_resnet18",
        "config": {
            "strategy": "fedavg",
            "dataset": "cifar10",
            "model": "resnet18",
            "partitioner": "iid",
            "client-selection": "random",
            "num-server-rounds": 20,
        },
    },

    # --- Dataset comparison ---
    {
        "name": "fedavg_mnist_iid_random",
        "config": {
            "strategy": "fedavg",
            "dataset": "mnist",
            "model": "cnn",
            "partitioner": "iid",
            "client-selection": "random",
            "num-server-rounds": 20,
        },
    },
    {
        "name": "fedavg_mnist_noniid_random",
        "config": {
            "strategy": "fedavg",
            "dataset": "mnist",
            "model": "cnn",
            "partitioner": "dirichlet",
            "dirichlet-alpha": 0.3,
            "client-selection": "random",
            "num-server-rounds": 20,
        },
    },

    # --- Differential Privacy ---
    {
        "name": "fedavg_cifar10_iid_dp",
        "config": {
            "strategy": "fedavg",
            "dataset": "cifar10",
            "model": "cnn",
            "partitioner": "iid",
            "client-selection": "random",
            "dp-clip": 1.0,
            "dp-noise": 0.01,
            "num-server-rounds": 20,
        },
    },
]


def build_run_config_str(config: dict) -> str:
    """Convert a config dict to a flwr --run-config string."""
    parts = []
    for k, v in config.items():
        if isinstance(v, str):
            parts.append(f"{k}='{v}'")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)


def run_experiment(name: str, config: dict, results_dir: Path, dry_run: bool = False):
    """Run a single experiment."""
    config_str = build_run_config_str(config)
    cmd = f'flwr run . --run-config "{config_str}"'

    print(f"\n{'='*60}")
    print(f"  Experiment: {name}")
    print(f"  Command: {cmd}")
    print(f"{'='*60}")

    if dry_run:
        print("  [DRY RUN] Skipped.")
        return None

    result = subprocess.run(cmd, shell=True, capture_output=False)

    # Move metrics.json to results dir with experiment name
    metrics_src = Path("metrics.json")
    if metrics_src.exists():
        dest = results_dir / f"metrics_{name}.json"
        shutil.move(str(metrics_src), str(dest))
        print(f"  Metrics saved to: {dest}")
        return dest
    else:
        print(f"  WARNING: metrics.json not found for {name}")
        return None


def print_comparison(results_dir: Path):
    """Print a comparison table of all experiment results."""
    files = sorted(results_dir.glob("metrics_*.json"))
    if not files:
        print("\nNo results to compare.")
        return

    print(f"\n{'='*100}")
    print("  EXPERIMENT COMPARISON")
    print(f"{'='*100}")
    header = f"{'Experiment':<40} {'Accuracy':>10} {'Best Acc':>10} {'Loss':>10} {'Comm(MB)':>10} {'Time(s)':>10}"
    print(header)
    print("-" * 100)

    for f in files:
        data = json.loads(f.read_text())
        name = f.stem.replace("metrics_", "")
        final_acc = data.get("final_accuracy", 0)
        best_acc = data.get("best_accuracy", 0)
        final_loss = data.get("final_loss", 0)
        comm_mb = data.get("total_communication_mb", 0)
        total_time = data.get("total_time_sec", 0)
        print(f"{name:<40} {final_acc:>10.4f} {best_acc:>10.4f} {final_loss:>10.4f} {comm_mb:>10.1f} {total_time:>10.1f}")

    print(f"{'='*100}")

    # Convergence comparison
    print(f"\n{'='*80}")
    print("  CONVERGENCE COMPARISON (rounds to reach target accuracy)")
    print(f"{'='*80}")
    targets = ["0.3", "0.5", "0.7", "0.8", "0.9"]
    header = f"{'Experiment':<40} " + " ".join(f"{t:>8}%" for t in targets)
    print(header)
    print("-" * 80)

    for f in files:
        data = json.loads(f.read_text())
        name = f.stem.replace("metrics_", "")
        conv = data.get("convergence_targets", {})
        vals = []
        for t in targets:
            r = conv.get(t, conv.get(float(t), None))
            vals.append(f"{r:>8}" if r is not None else f"{'N/A':>8}")
        print(f"{name:<40} " + " ".join(vals))

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Run batch FL experiments")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--filter", type=str, default="", help="Only run experiments whose name contains this string")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to store results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    experiments = EXPERIMENTS
    if args.filter:
        experiments = [e for e in experiments if args.filter.lower() in e["name"].lower()]

    print(f"Running {len(experiments)} experiment(s)...")
    print(f"Results directory: {results_dir.resolve()}")

    for exp in experiments:
        run_experiment(exp["name"], exp["config"], results_dir, dry_run=args.dry_run)

    if not args.dry_run:
        print_comparison(results_dir)


if __name__ == "__main__":
    main()
