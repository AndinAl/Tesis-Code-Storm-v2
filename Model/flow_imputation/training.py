from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .config import ImputationConfig
from .data import FlowImputationDataset
from .model import STGNNImputer


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float


def _candidate_window_starts(
    dataset: FlowImputationDataset,
    split: str,
    window_size: int,
    min_observations: int,
) -> np.ndarray:
    start_idx, end_idx = dataset.split_bounds(split)
    split_len = max(0, end_idx - start_idx)
    if split_len == 0:
        return np.array([], dtype=np.int64)

    effective_window = min(window_size, split_len)
    observed_counts = dataset.observed_mask[start_idx:end_idx].sum(dim=1).cpu().numpy()
    if effective_window == split_len:
        if observed_counts.sum() >= min_observations:
            return np.array([start_idx], dtype=np.int64)
        return np.array([], dtype=np.int64)

    kernel = np.ones(effective_window, dtype=np.int64)
    window_observations = np.convolve(observed_counts, kernel, mode="valid")
    starts = np.flatnonzero(window_observations >= min_observations) + start_idx
    return starts.astype(np.int64)


def _sample_holdout_mask(
    observed_mask: torch.Tensor,
    holdout_ratio: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if not bool(observed_mask.any()):
        return torch.zeros_like(observed_mask)

    random_draw = torch.rand(
        observed_mask.shape,
        generator=generator,
        device=observed_mask.device,
    )
    holdout = observed_mask & (random_draw < holdout_ratio)
    if bool(holdout.any()):
        return holdout

    observed_idx = torch.nonzero(observed_mask, as_tuple=False).squeeze(-1)
    sampled = observed_idx[
        torch.randint(len(observed_idx), (1,), generator=generator, device=observed_idx.device)
    ]
    holdout = torch.zeros_like(observed_mask)
    holdout[sampled] = True
    return holdout


def _snapshot_features(
    dataset: FlowImputationDataset,
    t: int,
    input_mask: torch.Tensor,
    normalized_flow: torch.Tensor,
    static_features: torch.Tensor,
    temporal_features: torch.Tensor,
) -> torch.Tensor:
    masked_flow = normalized_flow[t] * input_mask.float()
    temporal_block = temporal_features[t].unsqueeze(0).expand(dataset.num_segments, -1)
    return torch.cat(
        [
            masked_flow.unsqueeze(-1),
            input_mask.float().unsqueeze(-1),
            temporal_block,
            static_features,
        ],
        dim=-1,
    )


def _run_window(
    model: STGNNImputer,
    dataset: FlowImputationDataset,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    normalized_flow: torch.Tensor,
    observed_mask: torch.Tensor,
    static_features: torch.Tensor,
    temporal_features: torch.Tensor,
    window_start: int,
    window_size: int,
    holdout_ratio: float,
    generator: torch.Generator,
) -> torch.Tensor | None:
    hidden_state: torch.Tensor | None = None
    losses: list[torch.Tensor] = []
    max_t = min(window_start + window_size, dataset.num_steps)

    for t in range(window_start, max_t):
        observed_t = observed_mask[t]
        holdout_t = _sample_holdout_mask(observed_t, holdout_ratio, generator)
        input_mask = observed_t & ~holdout_t
        x = _snapshot_features(
            dataset=dataset,
            t=t,
            input_mask=input_mask,
            normalized_flow=normalized_flow,
            static_features=static_features,
            temporal_features=temporal_features,
        )
        prediction, hidden_state = model(x, edge_index, edge_weight, hidden_state)
        if bool(holdout_t.any()):
            losses.append(F.smooth_l1_loss(prediction[holdout_t], normalized_flow[t][holdout_t]))

    if not losses:
        return None
    return torch.stack(losses).mean()


def _device_string() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_imputer(
    model: STGNNImputer,
    dataset: FlowImputationDataset,
    split: str,
    cfg: ImputationConfig,
    device: str | None = None,
) -> float:
    device = device or _device_string()
    edge_index = dataset.edge_index.to(device)
    edge_weight = dataset.edge_weight.to(device)
    normalized_flow = dataset.normalized_flow.to(device)
    observed_mask = dataset.observed_mask.to(device)
    static_features = dataset.static_features.to(device)
    temporal_features = dataset.temporal_features.to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed + 10_000)

    starts = _candidate_window_starts(dataset, split, cfg.window_size, cfg.min_window_observations)
    if starts.size == 0:
        return float("nan")
    starts = starts[: min(len(starts), cfg.eval_windows)]

    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in starts:
            loss = _run_window(
                model=model,
                dataset=dataset,
                edge_index=edge_index,
                edge_weight=edge_weight,
                normalized_flow=normalized_flow,
                observed_mask=observed_mask,
                static_features=static_features,
                temporal_features=temporal_features,
                window_start=int(start),
                window_size=cfg.window_size,
                holdout_ratio=cfg.holdout_ratio,
                generator=generator,
            )
            if loss is not None:
                losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else float("nan")


def train_imputer(
    dataset: FlowImputationDataset,
    cfg: ImputationConfig | None = None,
) -> tuple[STGNNImputer, list[EpochMetrics]]:
    cfg = cfg or ImputationConfig()
    device = _device_string()
    model = STGNNImputer(
        node_features=dataset.feature_dim,
        hidden_dim=cfg.hidden_dim,
        K=cfg.cheb_k,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    edge_index = dataset.edge_index.to(device)
    edge_weight = dataset.edge_weight.to(device)
    normalized_flow = dataset.normalized_flow.to(device)
    observed_mask = dataset.observed_mask.to(device)
    static_features = dataset.static_features.to(device)
    temporal_features = dataset.temporal_features.to(device)

    rng = np.random.default_rng(cfg.seed)
    torch_generator = torch.Generator(device=device)
    torch_generator.manual_seed(cfg.seed)
    train_starts = _candidate_window_starts(
        dataset,
        split="train",
        window_size=cfg.window_size,
        min_observations=cfg.min_window_observations,
    )
    if train_starts.size == 0:
        raise RuntimeError("No train windows with observed data were found.")

    history: list[EpochMetrics] = []
    for epoch in range(cfg.epochs):
        sampled_starts = rng.choice(
            train_starts,
            size=min(len(train_starts), cfg.windows_per_epoch),
            replace=len(train_starts) < cfg.windows_per_epoch,
        )

        epoch_losses: list[float] = []
        model.train()
        for start in sampled_starts:
            optimizer.zero_grad()
            loss = _run_window(
                model=model,
                dataset=dataset,
                edge_index=edge_index,
                edge_weight=edge_weight,
                normalized_flow=normalized_flow,
                observed_mask=observed_mask,
                static_features=static_features,
                temporal_features=temporal_features,
                window_start=int(start),
                window_size=cfg.window_size,
                holdout_ratio=cfg.holdout_ratio,
                generator=torch_generator,
            )
            if loss is None:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        val_loss = evaluate_imputer(model, dataset, split="val", cfg=cfg, device=device)
        history.append(EpochMetrics(epoch=epoch, train_loss=train_loss, val_loss=val_loss))
        print(
            f"epoch={epoch:03d} train_loss={train_loss:8.5f} "
            f"val_loss={val_loss:8.5f}"
        )

    return model, history


def impute_full_flow_matrix(
    model: STGNNImputer,
    dataset: FlowImputationDataset,
    device: str | None = None,
    progress_every: int | None = 1000,
) -> torch.FloatTensor:
    device = device or _device_string()
    edge_index = dataset.edge_index.to(device)
    edge_weight = dataset.edge_weight.to(device)
    normalized_flow = dataset.normalized_flow.to(device)
    observed_mask = dataset.observed_mask.to(device)
    static_features = dataset.static_features.to(device)
    temporal_features = dataset.temporal_features.to(device)

    predictions = torch.zeros_like(normalized_flow)
    hidden_state: torch.Tensor | None = None
    model.eval()
    with torch.no_grad():
        for t in range(dataset.num_steps):
            if progress_every and t % progress_every == 0:
                print(f"imputing step {t}/{dataset.num_steps}")
            x = _snapshot_features(
                dataset=dataset,
                t=t,
                input_mask=observed_mask[t],
                normalized_flow=normalized_flow,
                static_features=static_features,
                temporal_features=temporal_features,
            )
            prediction, hidden_state = model(x, edge_index, edge_weight, hidden_state)
            predictions[t] = prediction
    if progress_every:
        print(f"imputing step {dataset.num_steps}/{dataset.num_steps}")

    completed_norm = predictions.cpu()
    completed_norm[dataset.observed_mask] = dataset.normalized_flow[dataset.observed_mask]
    completed_raw = torch.expm1(torch.clamp(completed_norm, min=0.0) * dataset.flow_scale)
    return completed_raw
