"""Custom federated learning strategies with configurable client selection."""

import random
import numpy as np
from collections import defaultdict
from typing import Iterable

from flwr.app import ArrayRecord, ConfigRecord, Message, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg, FedAdagrad


# ======================== Client Selection Strategies ========================

def select_clients_random(grid: Grid, fraction: float, min_nodes: int) -> list:
    """Random client selection (default Flower behavior).

    Randomly samples a fraction of available clients each round.
    """
    node_ids = list(grid.node_ids())
    num_to_select = max(min_nodes, int(len(node_ids) * fraction))
    num_to_select = min(num_to_select, len(node_ids))
    return random.sample(node_ids, num_to_select)


def select_clients_high_loss(
    grid: Grid, fraction: float, min_nodes: int, client_losses: dict
) -> list:
    """High-loss client selection.

    Prioritizes clients that had the highest training loss in the previous round,
    since they likely have the most to contribute to model improvement.
    """
    node_ids = list(grid.node_ids())
    num_to_select = max(min_nodes, int(len(node_ids) * fraction))
    num_to_select = min(num_to_select, len(node_ids))

    if not client_losses:
        return random.sample(node_ids, num_to_select)

    sorted_nodes = sorted(
        node_ids, key=lambda nid: client_losses.get(nid, 0.0), reverse=True
    )
    return sorted_nodes[:num_to_select]


def select_clients_cluster_based(
    grid: Grid,
    fraction: float,
    min_nodes: int,
    client_metrics: dict,
    num_clusters: int = 3,
) -> list:
    """Cluster-based client selection.

    Groups clients into clusters based on their local training metrics
    (loss, num-examples), then samples proportionally from each cluster
    to ensure diversity. This helps when data is non-IID — clients with
    similar data distributions are grouped together, and selecting from
    each cluster prevents bias toward one data distribution.

    Clustering uses simple K-Means on (loss, num_examples) feature vectors.

    Falls back to random selection in round 1 (no metrics available yet).
    """
    node_ids = list(grid.node_ids())
    num_to_select = max(min_nodes, int(len(node_ids) * fraction))
    num_to_select = min(num_to_select, len(node_ids))

    # Fallback: no prior metrics yet
    if not client_metrics:
        return random.sample(node_ids, num_to_select)

    # Build feature vectors: [loss, num_examples] for nodes with metrics
    nodes_with_metrics = [nid for nid in node_ids if nid in client_metrics]
    nodes_without_metrics = [nid for nid in node_ids if nid not in client_metrics]

    if len(nodes_with_metrics) < num_clusters:
        return random.sample(node_ids, num_to_select)

    features = np.array([
        [client_metrics[nid].get("loss", 0.0), client_metrics[nid].get("num_examples", 0)]
        for nid in nodes_with_metrics
    ])

    # Normalize features to [0, 1] for fair distance computation
    f_min = features.min(axis=0)
    f_max = features.max(axis=0)
    denom = f_max - f_min
    denom[denom == 0] = 1.0
    features_norm = (features - f_min) / denom

    # Simple K-Means clustering
    actual_k = min(num_clusters, len(nodes_with_metrics))
    labels = _kmeans(features_norm, actual_k)

    # Group nodes by cluster
    clusters = defaultdict(list)
    for i, nid in enumerate(nodes_with_metrics):
        clusters[labels[i]].append(nid)

    # Proportional sampling from each cluster
    selected = []
    remaining_budget = num_to_select
    cluster_ids = sorted(clusters.keys())

    for idx, cid in enumerate(cluster_ids):
        members = clusters[cid]
        if idx == len(cluster_ids) - 1:
            # Last cluster gets whatever budget is left
            n_from_cluster = remaining_budget
        else:
            n_from_cluster = max(1, int(num_to_select * len(members) / len(nodes_with_metrics)))
        n_from_cluster = min(n_from_cluster, len(members), remaining_budget)
        selected.extend(random.sample(members, n_from_cluster))
        remaining_budget -= n_from_cluster
        if remaining_budget <= 0:
            break

    # If still under budget, fill from nodes without metrics or remaining nodes
    if remaining_budget > 0:
        pool = [nid for nid in node_ids if nid not in selected]
        fill = min(remaining_budget, len(pool))
        selected.extend(random.sample(pool, fill))

    return selected[:num_to_select]


def _kmeans(X: np.ndarray, k: int, max_iter: int = 20) -> list:
    """Lightweight K-Means (no sklearn dependency)."""
    n = len(X)
    # Initialize centroids with K-Means++ style
    indices = [random.randint(0, n - 1)]
    for _ in range(1, k):
        dists = np.min([np.sum((X - X[c]) ** 2, axis=1) for c in indices], axis=0)
        probs = dists / (dists.sum() + 1e-12)
        indices.append(np.random.choice(n, p=probs))
    centroids = X[indices].copy()

    labels = [0] * n
    for _ in range(max_iter):
        # Assign
        for i in range(n):
            dists = [np.sum((X[i] - centroids[j]) ** 2) for j in range(k)]
            labels[i] = int(np.argmin(dists))
        # Update
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(k)
        for i in range(n):
            new_centroids[labels[i]] += X[i]
            counts[labels[i]] += 1
        for j in range(k):
            if counts[j] > 0:
                new_centroids[j] /= counts[j]
            else:
                new_centroids[j] = centroids[j]
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels


def select_clients_power_of_choice(
    grid: Grid, fraction: float, min_nodes: int, client_losses: dict, d: int = 2
) -> list:
    """Power-of-d-choice client selection.

    For each slot, randomly sample d candidate clients and pick the one with
    the highest loss. Balances exploration and exploitation.
    """
    node_ids = list(grid.node_ids())
    num_to_select = max(min_nodes, int(len(node_ids) * fraction))
    num_to_select = min(num_to_select, len(node_ids))

    if not client_losses or len(node_ids) <= num_to_select:
        return random.sample(node_ids, num_to_select)

    selected = []
    remaining = list(node_ids)
    for _ in range(num_to_select):
        candidates = random.sample(remaining, min(d, len(remaining)))
        best = max(candidates, key=lambda nid: client_losses.get(nid, 0.0))
        selected.append(best)
        remaining.remove(best)
    return selected


# ======================== Strategy: Custom FedAvg ========================

class CustomFedAvg(FedAvg):
    """FedAvg with configurable client selection strategy."""

    def __init__(self, client_selection: str = "random", **kwargs):
        super().__init__(**kwargs)
        self.client_selection = client_selection
        self.client_losses = {}
        self.client_metrics = {}

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        print(f"[Round {server_round}] Strategy=FedAvg, ClientSelection={self.client_selection}")
        return super().configure_train(server_round, arrays, config, grid)


# ======================== Strategy: Custom FedAdagrad ========================

class CustomFedAdagrad(FedAdagrad):
    """FedAdagrad with LR decay and configurable client selection strategy."""

    def __init__(
        self,
        client_selection: str = "random",
        lr_decay_interval: int = 5,
        lr_decay_factor: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.client_selection = client_selection
        self.client_losses = {}
        self.client_metrics = {}
        self.lr_decay_interval = lr_decay_interval
        self.lr_decay_factor = lr_decay_factor

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        if server_round % self.lr_decay_interval == 0 and server_round > 0:
            config["lr"] *= self.lr_decay_factor
            print(f"[Round {server_round}] LR decreased to: {config['lr']}")
        print(f"[Round {server_round}] Strategy=FedAdagrad, ClientSelection={self.client_selection}")
        return super().configure_train(server_round, arrays, config, grid)
