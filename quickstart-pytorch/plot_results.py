"""Visualization utility for federated learning experiment results.

Reads metrics_*.json files from the results directory and generates
comparison plots for accuracy, loss, convergence, and communication cost.

Usage:
    python plot_results.py                          # default: ./results
    python plot_results.py --results-dir ./results  # specify directory
    python plot_results.py --filter noniid          # only plot matching experiments
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: Path, name_filter: str = "") -> dict:
    """Load all metrics_*.json files from the results directory."""
    results = {}
    for f in sorted(results_dir.glob("metrics_*.json")):
        name = f.stem.replace("metrics_", "")
        if name_filter and name_filter.lower() not in name.lower():
            continue
        data = json.loads(f.read_text())
        if data:
            results[name] = data
    return results


def plot_accuracy_curves(results: dict, output_dir: Path):
    """Plot per-round accuracy curves for all experiments."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in results.items():
        rounds = [r["round"] for r in data["per_round"]]
        accs = [r["accuracy"] for r in data["per_round"]]
        ax.plot(rounds, accs, marker="o", markersize=3, label=name)
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Global Test Accuracy", fontsize=12)
    ax.set_title("Accuracy vs. Communication Round", fontsize=14)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_curves.png", dpi=150)
    print(f"  Saved: {output_dir / 'accuracy_curves.png'}")
    plt.close(fig)


def plot_loss_curves(results: dict, output_dir: Path):
    """Plot per-round loss curves for all experiments."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in results.items():
        rounds = [r["round"] for r in data["per_round"]]
        losses = [r["loss"] for r in data["per_round"]]
        ax.plot(rounds, losses, marker="o", markersize=3, label=name)
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Global Test Loss", fontsize=12)
    ax.set_title("Loss vs. Communication Round", fontsize=14)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=150)
    print(f"  Saved: {output_dir / 'loss_curves.png'}")
    plt.close(fig)


def plot_accuracy_vs_communication(results: dict, output_dir: Path):
    """Plot accuracy as a function of cumulative communication cost (MB)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in results.items():
        comm = [r["cumulative_comm_mb"] for r in data["per_round"]]
        accs = [r["accuracy"] for r in data["per_round"]]
        ax.plot(comm, accs, marker="o", markersize=3, label=name)
    ax.set_xlabel("Cumulative Communication Cost (MB)", fontsize=12)
    ax.set_ylabel("Global Test Accuracy", fontsize=12)
    ax.set_title("Accuracy vs. Communication Cost", fontsize=14)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_comm.png", dpi=150)
    print(f"  Saved: {output_dir / 'accuracy_vs_comm.png'}")
    plt.close(fig)


def plot_convergence_bar(results: dict, output_dir: Path):
    """Bar chart: rounds to reach target accuracies."""
    targets = [0.3, 0.5, 0.7, 0.8]
    names = list(results.keys())
    x = np.arange(len(targets))
    width = 0.8 / max(len(names), 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(names):
        conv = results[name].get("convergence_targets", {})
        vals = []
        for t in targets:
            r = conv.get(str(t), conv.get(t, None))
            vals.append(r if r is not None else 0)
        bars = ax.bar(x + i * width, vals, width, label=name)
        # Mark N/A
        for j, v in enumerate(vals):
            if v == 0:
                ax.text(x[j] + i * width, 0.5, "N/A", ha="center", va="bottom", fontsize=7, color="red")

    ax.set_xlabel("Target Accuracy", fontsize=12)
    ax.set_ylabel("Rounds to Reach", fontsize=12)
    ax.set_title("Convergence Speed Comparison", fontsize=14)
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels([f"{int(t*100)}%" for t in targets])
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "convergence_bar.png", dpi=150)
    print(f"  Saved: {output_dir / 'convergence_bar.png'}")
    plt.close(fig)


def plot_final_comparison_bar(results: dict, output_dir: Path):
    """Bar chart comparing final accuracy, communication cost, and time."""
    names = list(results.keys())
    final_accs = [results[n]["final_accuracy"] for n in names]
    comm_mbs = [results[n]["total_communication_mb"] for n in names]
    total_times = [results[n]["total_time_sec"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Final accuracy
    axes[0].barh(names, final_accs, color="steelblue")
    axes[0].set_xlabel("Final Accuracy")
    axes[0].set_title("Final Accuracy")
    axes[0].set_xlim(0, 1)
    for i, v in enumerate(final_accs):
        axes[0].text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=8)

    # Communication cost
    axes[1].barh(names, comm_mbs, color="coral")
    axes[1].set_xlabel("Communication Cost (MB)")
    axes[1].set_title("Total Communication Cost")
    for i, v in enumerate(comm_mbs):
        axes[1].text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=8)

    # Total time
    axes[2].barh(names, total_times, color="mediumseagreen")
    axes[2].set_xlabel("Total Time (s)")
    axes[2].set_title("Total Training Time")
    for i, v in enumerate(total_times):
        axes[2].text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=8)

    fig.suptitle("Experiment Comparison Summary", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "final_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_dir / 'final_comparison.png'}")
    plt.close(fig)


def plot_round_time(results: dict, output_dir: Path):
    """Plot per-round training time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in results.items():
        rounds = [r["round"] for r in data["per_round"]]
        times = [r["round_time_sec"] for r in data["per_round"]]
        ax.plot(rounds, times, marker="o", markersize=3, label=name)
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Round Time (seconds)", fontsize=12)
    ax.set_title("Per-Round Training Time", fontsize=14)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "round_time.png", dpi=150)
    print(f"  Saved: {output_dir / 'round_time.png'}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot FL experiment results")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing metrics_*.json")
    parser.add_argument("--output-dir", type=str, default="", help="Directory for plots (default: same as results-dir)")
    parser.add_argument("--filter", type=str, default="", help="Only plot experiments whose name contains this string")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(exist_ok=True)

    results = load_results(results_dir, args.filter)
    if not results:
        print(f"No metrics files found in {results_dir}. Run experiments first.")
        return

    print(f"Found {len(results)} experiment(s): {', '.join(results.keys())}")
    print(f"Generating plots in {output_dir.resolve()}...\n")

    plot_accuracy_curves(results, output_dir)
    plot_loss_curves(results, output_dir)
    plot_accuracy_vs_communication(results, output_dir)
    plot_convergence_bar(results, output_dir)
    plot_final_comparison_bar(results, output_dir)
    plot_round_time(results, output_dir)

    print(f"\nAll plots saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
