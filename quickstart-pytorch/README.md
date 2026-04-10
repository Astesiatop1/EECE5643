# Federated Learning Simulation Platform

A configurable federated learning simulation platform based on Flower and PyTorch. Supports multiple FL strategies (FedAvg, FedProx, FedAdagrad), datasets, models, client selection algorithms, and differential privacy, with comprehensive evaluation metrics and automated experiment tooling.

## Project Structure

```
quickstart-pytorch/
├── pytorchexample/
│   ├── __init__.py
│   ├── client_app.py        # ClientApp: local training & evaluation
│   ├── server_app.py        # ServerApp: orchestration & global evaluation
│   ├── task.py              # Model definitions, dataset loading, train/test
│   ├── custom_strategy.py   # Custom FL strategies & client selection algorithms
│   └── metrics_tracker.py   # Evaluation metrics: accuracy, convergence, comm cost
├── run_experiments.py        # Batch experiment runner (run all configs at once)
├── plot_results.py           # Visualization: generate comparison plots
├── pyproject.toml            # Dependencies & default run config
└── README.md
```

## Environment Setup

### Prerequisites

- Python 3.10+
- pip
- (Optional) CUDA-compatible GPU for acceleration

### Step 1: Create Virtual Environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -e .
```

This will install all required packages defined in `pyproject.toml`:

| Package | Version | Purpose |
|---------|---------|---------|
| `flwr[simulation]` | >= 1.26.0 | Flower framework with simulation engine |
| `flwr-datasets[vision]` | >= 0.5.0 | Federated dataset partitioning |
| `torch` | 2.8.0 | PyTorch deep learning framework |
| `torchvision` | 0.23.0 | Pre-trained models (ResNet18) & transforms |
| `numpy` | latest | Numerical computing (K-Means clustering) |
| `matplotlib` | latest | Result visualization and plotting |

> If you need a different PyTorch version (e.g. CUDA-specific), install it first following https://pytorch.org/get-started/locally/, then run `pip install -e .`.

## How to Run

### Basic Run (default config)

```bash
flwr run .
```

This uses all default values from `pyproject.toml`: FedAvg + Simple CNN + CIFAR-10 + IID + Random selection.

### Run with Custom Config

Use `--run-config` to override any parameter at runtime:

```bash
flwr run . --run-config "key1=value1 key2=value2"
```

## Configurable Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `strategy` | `fedavg`, `fedprox`, `fedadagrad` | `fedavg` | FL aggregation strategy |
| `proximal-mu` | float | `0.1` | FedProx proximal term coefficient (only for `fedprox`) |
| `model` | `cnn`, `resnet18` | `cnn` | Model architecture |
| `dataset` | `cifar10`, `mnist`, `fashion-mnist` | `cifar10` | Training dataset |
| `partitioner` | `iid`, `dirichlet` | `iid` | Data partition method |
| `dirichlet-alpha` | float (e.g. 0.1 ~ 1.0) | `0.5` | Non-IID degree (lower = more heterogeneous) |
| `client-selection` | `random`, `high-loss`, `cluster-based`, `power-of-choice` | `random` | Client selection algorithm |
| `num-server-rounds` | int | `3` | Number of FL rounds |
| `learning-rate` | float | `0.1` | Learning rate |
| `local-epochs` | int | `1` | Local training epochs per round |
| `batch-size` | int | `32` | Batch size |
| `fraction-evaluate` | float (0~1) | `0.5` | Fraction of clients for evaluation |
| `franction-train` | float (0~1) | `0.5` | Fraction of clients for training |
| `dp-clip` | float (0 = off) | `0.0` | Differential privacy: gradient clipping max L2 norm |
| `dp-noise` | float (0 = off) | `0.0` | Differential privacy: Gaussian noise std |

## Example Experiments

### Single Experiment

```bash
# FedAvg + CIFAR-10 + IID
flwr run . --run-config "strategy='fedavg' dataset='cifar10' partitioner='iid' client-selection='random' model='cnn' num-server-rounds=20"

# FedProx + CIFAR-10 + Non-IID (Dirichlet)
flwr run . --run-config "strategy='fedprox' proximal-mu=0.1 dataset='cifar10' partitioner='dirichlet' dirichlet-alpha=0.3 client-selection='random' num-server-rounds=20"

# FedAvg + Non-IID + Cluster-based selection
flwr run . --run-config "strategy='fedavg' dataset='cifar10' partitioner='dirichlet' dirichlet-alpha=0.3 client-selection='cluster-based' num-server-rounds=20"

# FedAvg + Differential Privacy enabled
flwr run . --run-config "strategy='fedavg' dataset='cifar10' dp-clip=1.0 dp-noise=0.01 num-server-rounds=20"

# ResNet18 + MNIST + Non-IID
flwr run . --run-config "strategy='fedadagrad' dataset='mnist' model='resnet18' partitioner='dirichlet' dirichlet-alpha=0.5 client-selection='high-loss' num-server-rounds=20"
```

### Batch Experiments

Run all predefined experiment combinations at once:

```bash
# Run all experiments, save results to ./results/
python run_experiments.py

# Preview commands without running
python run_experiments.py --dry-run

# Run only experiments matching a keyword
python run_experiments.py --filter noniid

# Custom results directory
python run_experiments.py --results-dir ./my_results
```

The predefined experiments in `run_experiments.py` cover:
- Strategy comparison (FedAvg vs FedProx vs FedAdagrad) under IID and non-IID
- Client selection comparison (Random vs High-loss vs Cluster-based vs Power-of-choice)
- Model comparison (Simple CNN vs ResNet18)
- Dataset comparison (CIFAR-10 vs MNIST)
- Differential privacy impact

### Visualization

After running experiments, generate comparison plots:

```bash
# Generate all plots from results/
python plot_results.py

# Only plot specific experiments
python plot_results.py --filter cifar10

# Custom directories
python plot_results.py --results-dir ./results --output-dir ./plots
```

Generated plots:
- `accuracy_curves.png` — accuracy vs. round for all experiments
- `loss_curves.png` — loss vs. round
- `accuracy_vs_comm.png` — accuracy vs. cumulative communication cost (MB)
- `convergence_bar.png` — rounds to reach 30%/50%/70%/80% accuracy
- `final_comparison.png` — side-by-side bar chart of final accuracy, comm cost, and time
- `round_time.png` — per-round training time

## Output

After each experiment:

- **Console**: prints per-round metrics and a final summary table
- **`metrics.json`**: full experiment results including:
  - Per-round accuracy, loss, accuracy delta
  - Convergence speed (rounds to reach 30%/50%/70%/80%/90% accuracy)
  - Communication cost per round and cumulative (in MB)
  - Round time and total time
  - Experiment config snapshot
- **`final_model.pt`**: saved final global model weights

## Client Count Configuration

Change the number of simulated clients in `pyproject.toml`:

```toml
[tool.flwr.federations.local-simulation]
options.num-supernodes = 10  # number of simulated clients
```

## GPU Acceleration

If your system has a CUDA GPU, the code will automatically use it. To configure GPU resources for simulation, edit `pyproject.toml`:

```toml
[tool.flwr.federations.local-simulation.options]
backend.client-resources.num-gpus = 0.5  # GPU fraction per client
```
