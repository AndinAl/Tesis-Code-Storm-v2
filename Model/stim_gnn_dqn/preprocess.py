from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from .data_loader import STIMDataset


@dataclass
class StaticParameters:
    capacity: torch.FloatTensor
    inbound_capacity: torch.FloatTensor
    thresholds: torch.FloatTensor
    capacity_norm: torch.FloatTensor
    threshold_norm: torch.FloatTensor
    distance_norm: torch.FloatTensor
    coords_norm: torch.FloatTensor
    indegree_norm: torch.FloatTensor
    outdegree_norm: torch.FloatTensor
    community_norm: torch.FloatTensor
    gateway_norm: torch.FloatTensor
    train_starts: List[int]
    val_starts: List[int]
    test_starts: List[int]

    @property
    def betweenness_norm(self) -> torch.FloatTensor:
        return self.gateway_norm


def _normalize(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    denom = torch.clamp(x.max() - x.min(), min=1e-6)
    return (x - x.min()) / denom


def build_static_parameters(
    dataset: STIMDataset,
    horizon: int,
    train_ratio: float,
    val_ratio: float,
    q: float,
    threshold_alpha: float,
    validation_start: str | None = None,
    validation_end: str | None = None,
) -> StaticParameters:
    full_days = dataset.num_steps // horizon
    all_starts = [d * horizon for d in range(full_days)]

    use_date_window = (
        dataset.timestamps is not None
        and validation_start is not None
        and validation_end is not None
    )

    if use_date_window:
        timestamps = dataset.timestamps
        val_start = np.datetime64(validation_start)
        val_end = np.datetime64(validation_end)

        train_starts = []
        val_starts = []
        test_starts = []
        for start in all_starts:
            episode_start = timestamps[start]
            episode_end = timestamps[min(start + horizon - 1, dataset.num_steps - 1)]
            overlaps_val = episode_start <= val_end and episode_end >= val_start
            if overlaps_val:
                val_starts.append(start)
            elif episode_end < val_start:
                train_starts.append(start)
            else:
                test_starts.append(start)

        # Fallback if the provided window does not intersect the data.
        if not val_starts or not train_starts:
            use_date_window = False

    if not use_date_window:
        n_train = max(1, int(full_days * train_ratio))
        n_val = max(1, int(full_days * val_ratio))
        if n_train + n_val >= full_days:
            n_val = max(1, full_days - n_train - 1)

        train_starts = all_starts[:n_train]
        val_starts = all_starts[n_train : n_train + n_val]
        test_starts = all_starts[n_train + n_val :]

    train_time_idx: list[int] = []
    for start in train_starts:
        end = min(start + horizon, dataset.num_steps)
        train_time_idx.extend(range(start, end))
    if not train_time_idx:
        train_time_idx = list(range(min(horizon, dataset.num_steps)))
    train_flows = dataset.flows[torch.tensor(train_time_idx, dtype=torch.long)]

    capacity = torch.quantile(train_flows, q=q, dim=0)
    capacity = torch.clamp(capacity, min=0.0)

    src, dst = dataset.edge_index
    if src.numel() != dataset.num_edges or dst.numel() != dataset.num_edges:
        raise ValueError(
            "edge_index shape is incompatible with directed edge flows: "
            f"num_edges={dataset.num_edges} src={src.numel()} dst={dst.numel()}"
        )
    inbound_capacity = torch.zeros(dataset.num_nodes, dtype=torch.float32)
    # Directed aggregation on destination nodes preserves asymmetric
    # sink/source behavior across opposite carriageways.
    inbound_capacity.index_add_(0, dst, capacity)

    thresholds = threshold_alpha * inbound_capacity

    coords_norm = torch.stack(
        [_normalize(dataset.coords[:, 0]), _normalize(dataset.coords[:, 1])], dim=1
    )

    return StaticParameters(
        capacity=capacity,
        inbound_capacity=inbound_capacity,
        thresholds=thresholds,
        capacity_norm=_normalize(capacity),
        threshold_norm=_normalize(thresholds),
        distance_norm=_normalize(dataset.distances),
        coords_norm=coords_norm,
        indegree_norm=_normalize(dataset.indegree),
        outdegree_norm=_normalize(dataset.outdegree),
        community_norm=_normalize(dataset.community_ids.float()),
        gateway_norm=_normalize(dataset.betweenness_centrality),
        train_starts=train_starts,
        val_starts=val_starts,
        test_starts=test_starts,
    )
