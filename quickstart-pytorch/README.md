# Federated Learning Simulation Platform

A configurable federated learning simulation platform based on Flower and PyTorch. Supports multiple FL strategies, datasets, models, and client selection algorithms, with comprehensive evaluation metrics tracking.

## Project Structure

```
quickstart-pytorch
├── pytorchexample
│   ├── __init__.py
│   ├── client_app.py        # ClientApp: local training & evaluation
│   ├── server_app.py        # ServerApp: orchestration & global evaluation
│   ├── task.py              # Model definitions, dataset loading, train/test
│   ├── custom_strategy.py   # Custom FL strategies & client selection algorithms
│   └── metrics_tracker.py   # Evaluation metrics: accuracy, convergence, comm cost
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
| `strategy` | `fedavg`, `fedadagrad` | `fedavg` | FL aggregation strategy |
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

## Example Experiments

```bash
# Exp 1: CIFAR-10, IID, FedAvg, Random, Simple CNN
flwr run . --run-config "dataset='cifar10' partitioner='iid' strategy='fedavg' client-selection='random' model='cnn' num-server-rounds=20"

# Exp 2: CIFAR-10, Non-IID, FedAvg, Cluster-based, Simple CNN
flwr run . --run-config "dataset='cifar10' partitioner='dirichlet' dirichlet-alpha=0.3 strategy='fedavg' client-selection='cluster-based' model='cnn' num-server-rounds=20"

# Exp 3: MNIST, Non-IID, FedAdagrad, High-loss, ResNet18
flwr run . --run-config "dataset='mnist' partitioner='dirichlet' dirichlet-alpha=0.5 strategy='fedadagrad' client-selection='high-loss' model='resnet18' num-server-rounds=20"

# Exp 4: Fashion-MNIST, Non-IID, FedAvg, Power-of-choice, CNN
flwr run . --run-config "dataset='fashion-mnist' partitioner='dirichlet' dirichlet-alpha=0.1 strategy='fedavg' client-selection='power-of-choice' model='cnn' num-server-rounds=15"
```

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

## GPU Acceleration

If your system has a CUDA GPU, the code will automatically use it. To configure GPU resources for simulation, edit `pyproject.toml`:

```toml
[tool.flwr.app.config]
# ... existing config ...

[tool.flwr.federations.local-simulation.options]
backend.client-resources.num-gpus = 0.5  # GPU fraction per client
```

Or run with:

```bash
flwr run . --run-config "num-server-rounds=20" --federation local-simulation
```
