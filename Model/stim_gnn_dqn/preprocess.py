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
    # Residual baseline tensor mu[h, d] per directed edge:
    # shape [2 (weekday/weekend), 24 hours, num_edges].
    residual_mu_by_day_hour: torch.FloatTensor
    # Cached timestamp decomposition for fast residual lookup during rollouts.
    time_hour_idx: torch.LongTensor
    time_daytype_idx: torch.LongTensor
    # Time-regime one-hot vectors per timestep:
    # [early_morning, morning_peak, midday, evening_peak, night].
    time_regime_onehot: torch.FloatTensor
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


def _extract_hour_daytype(
    dataset: STIMDataset,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-step hour index and day-type index (0 weekday, 1 weekend)."""
    if dataset.timestamps is None:
        hour_idx = (np.arange(dataset.num_steps, dtype=np.int64) % max(1, horizon)).astype(np.int64)
        daytype_idx = np.zeros(dataset.num_steps, dtype=np.int64)
        return hour_idx, daytype_idx

    ts_hour = np.asarray(dataset.timestamps).astype("datetime64[h]")
    ts_day = np.asarray(dataset.timestamps).astype("datetime64[D]")
    # 1970-01-01 is Thursday -> Monday=0 conversion offset is +3.
    dow = (ts_day.astype(np.int64) + 3) % 7
    daytype_idx = (dow >= 5).astype(np.int64)  # 0 weekday, 1 weekend
    hour_idx = (ts_hour.astype(np.int64) % 24).astype(np.int64)
    return hour_idx, daytype_idx


def _compute_residual_mu_by_day_hour(
    dataset: STIMDataset,
    train_time_idx: list[int],
    hour_idx: np.ndarray,
    daytype_idx: np.ndarray,
) -> torch.FloatTensor:
    """Compute mu(h,d) baseline per directed edge from training timesteps."""
    flows_np = dataset.flows.detach().cpu().numpy().astype(np.float32)
    num_edges = dataset.num_edges
    mu = np.zeros((2, 24, num_edges), dtype=np.float64)
    cnt = np.zeros((2, 24), dtype=np.int64)

    for t in train_time_idx:
        d = int(daytype_idx[t])
        h = int(hour_idx[t])
        mu[d, h] += flows_np[t]
        cnt[d, h] += 1

    global_edge_mean = flows_np[train_time_idx].mean(axis=0) if train_time_idx else flows_np.mean(axis=0)
    for d in range(2):
        for h in range(24):
            if cnt[d, h] > 0:
                mu[d, h] /= float(cnt[d, h])
            else:
                mu[d, h] = global_edge_mean

    return torch.tensor(mu.astype(np.float32), dtype=torch.float32)


def _compute_time_regime_onehot(hour_idx: np.ndarray) -> torch.FloatTensor:
    """Map hour -> regime and return one-hot [num_steps, 5]."""
    regime_idx = np.full(hour_idx.shape, 4, dtype=np.int64)  # default night
    regime_idx[(hour_idx >= 0) & (hour_idx <= 5)] = 0  # early_morning
    regime_idx[(hour_idx >= 6) & (hour_idx <= 9)] = 1  # morning_peak
    regime_idx[(hour_idx >= 10) & (hour_idx <= 15)] = 2  # midday
    regime_idx[(hour_idx >= 16) & (hour_idx <= 19)] = 3  # evening_peak
    regime_idx[(hour_idx >= 20) & (hour_idx <= 23)] = 4  # night
    onehot = np.eye(5, dtype=np.float32)[regime_idx]
    return torch.tensor(onehot, dtype=torch.float32)


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

    hour_idx, daytype_idx = _extract_hour_daytype(dataset=dataset, horizon=horizon)
    residual_mu = _compute_residual_mu_by_day_hour(
        dataset=dataset,
        train_time_idx=train_time_idx,
        hour_idx=hour_idx,
        daytype_idx=daytype_idx,
    )
    time_regime_onehot = _compute_time_regime_onehot(hour_idx=hour_idx)

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
        residual_mu_by_day_hour=residual_mu,
        time_hour_idx=torch.tensor(hour_idx, dtype=torch.long),
        time_daytype_idx=torch.tensor(daytype_idx, dtype=torch.long),
        time_regime_onehot=time_regime_onehot,
        train_starts=train_starts,
        val_starts=val_starts,
        test_starts=test_starts,
    )
