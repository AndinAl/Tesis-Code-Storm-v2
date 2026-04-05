from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import FlowImputationDataset


def export_imputed_flow_matrix(
    dataset: FlowImputationDataset,
    completed_flow: torch.FloatTensor,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    np.savez_compressed(
        output_path,
        flow=completed_flow.cpu().numpy().astype(np.float32),
        observed_mask=dataset.observed_mask.cpu().numpy().astype(np.uint8),
        timestamps=dataset.timestamps.astype(str).to_numpy(),
        source=np.array([segment.source for segment in dataset.directed_segments]),
        target=np.array([segment.target for segment in dataset.directed_segments]),
        direction=np.array([segment.direction for segment in dataset.directed_segments]),
        br=np.array([segment.br for segment in dataset.directed_segments], dtype=np.int32),
        km_start=np.array([segment.km_start for segment in dataset.directed_segments], dtype=np.float32),
        km_end=np.array([segment.km_end for segment in dataset.directed_segments], dtype=np.float32),
        edge_index=dataset.edge_index.cpu().numpy(),
        edge_weight=dataset.edge_weight.cpu().numpy(),
    )


def export_segment_metadata(
    dataset: FlowImputationDataset,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    metadata = pd.DataFrame(
        {
            "segment_id": [segment.segment_id for segment in dataset.directed_segments],
            "edge_idx": [segment.edge_idx for segment in dataset.directed_segments],
            "source": [segment.source for segment in dataset.directed_segments],
            "target": [segment.target for segment in dataset.directed_segments],
            "direction": [segment.direction for segment in dataset.directed_segments],
            "br": [segment.br for segment in dataset.directed_segments],
            "km_start": [segment.km_start for segment in dataset.directed_segments],
            "km_end": [segment.km_end for segment in dataset.directed_segments],
            "length_km": [segment.length_km for segment in dataset.directed_segments],
            "midpoint_lat": [segment.midpoint_lat for segment in dataset.directed_segments],
            "midpoint_lon": [segment.midpoint_lon for segment in dataset.directed_segments],
        }
    )
    metadata.to_csv(output_path, index=False)


def build_snapshot_dict(
    dataset: FlowImputationDataset,
    completed_flow: torch.FloatTensor,
    max_steps: int | None = None,
) -> dict[str, dict[str, float]]:
    step_limit = dataset.num_steps if max_steps is None else min(dataset.num_steps, max_steps)
    snapshots: dict[str, dict[str, float]] = {}
    flow_np = completed_flow.cpu().numpy()
    for t in range(step_limit):
        snapshot = {}
        for segment in dataset.directed_segments:
            key = str((segment.source, segment.target))
            snapshot[key] = float(flow_np[t, segment.segment_id])
        snapshots[str(dataset.timestamps[t])] = snapshot
    return snapshots
