from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .environment import (
    incident_amplitude_to_directions,
    incident_text_to_directions,
    normalize_incident_amplitude,
)


@dataclass
class IncidentMappingResult:
    mapped: pd.DataFrame
    unmapped: pd.DataFrame

    @property
    def mapped_rows(self) -> int:
        return int(len(self.mapped))

    @property
    def unmapped_rows(self) -> int:
        return int(len(self.unmapped))


def _amplitude_factor(amplitude: object, both_factor: float, single_factor: float) -> float:
    text = normalize_incident_amplitude(amplitude)
    if "AMBOS" in text:
        return float(both_factor)
    return float(single_factor)


def map_incidents_to_segments(
    incidents_xlsx: str,
    metadata_csv: str,
    both_factor: float,
    single_factor: float,
) -> IncidentMappingResult:
    incidents = pd.read_excel(incidents_xlsx).copy()
    meta = pd.read_csv(metadata_csv).copy()

    incidents["data_hora_inicial"] = pd.to_datetime(incidents["data_hora_inicial"], errors="coerce")
    incidents["data_hora_final"] = pd.to_datetime(incidents["data_hora_final"], errors="coerce")
    incidents = incidents.dropna(subset=["data_hora_inicial", "data_hora_final", "br", "km"]).copy()

    mapped_rows: list[dict[str, object]] = []
    unmapped_rows: list[dict[str, object]] = []
    direction_hint_cols = [
        col
        for col in incidents.columns
        if ("sentido" in str(col).lower() or "dire" in str(col).lower())
        and str(col).lower() != "amplitude"
    ]

    for row in incidents.to_dict(orient="records"):
        br = int(row["br"])
        km = float(row["km"])
        amplitude = row.get("Amplitude")
        dirs = incident_amplitude_to_directions(amplitude)
        for col in direction_hint_cols:
            parsed = incident_text_to_directions(row.get(col))
            if parsed is not None:
                dirs = dirs.intersection(parsed)
        if not dirs:
            dirs = incident_amplitude_to_directions(amplitude)
        factor = _amplitude_factor(amplitude, both_factor=both_factor, single_factor=single_factor)

        segment_matches = meta[
            (meta["br"] == br)
            & (meta["km_start"] <= km)
            & (meta["km_end"] >= km)
            & (meta["direction"].isin(dirs))
        ]

        if segment_matches.empty:
            unmapped_rows.append(
                {
                    "data_hora_inicial": row["data_hora_inicial"],
                    "data_hora_final": row["data_hora_final"],
                    "br": br,
                    "km": km,
                    "Amplitude": amplitude,
                }
            )
            continue

        for segment in segment_matches.itertuples(index=False):
            mapped_rows.append(
                {
                    "data_hora_inicial": row["data_hora_inicial"],
                    "data_hora_final": row["data_hora_final"],
                    "br": br,
                    "km": km,
                    "Amplitude": amplitude,
                    "segment_id": int(segment.segment_id),
                    "direction": str(segment.direction),
                    "capacity_factor": factor,
                }
            )

    mapped = pd.DataFrame(mapped_rows)
    unmapped = pd.DataFrame(unmapped_rows)
    return IncidentMappingResult(mapped=mapped, unmapped=unmapped)


def build_incident_schedule_for_episode(
    start_t: int,
    horizon: int,
    num_edges: int,
    timestamps: np.ndarray,
    mapped_incidents: pd.DataFrame,
) -> torch.FloatTensor:
    schedule = np.ones((horizon, num_edges), dtype=np.float32)
    if mapped_incidents.empty:
        return torch.tensor(schedule, dtype=torch.float32)

    episode_ts = pd.to_datetime(timestamps[start_t : start_t + horizon])
    if len(episode_ts) == 0:
        return torch.tensor(schedule, dtype=torch.float32)

    for row in mapped_incidents.itertuples(index=False):
        start = pd.Timestamp(row.data_hora_inicial)
        end = pd.Timestamp(row.data_hora_final)
        if end < episode_ts[0] or start > episode_ts[-1]:
            continue
        active_mask = (episode_ts >= start) & (episode_ts <= end)
        if not bool(active_mask.any()):
            continue
        step_ids = np.where(active_mask)[0]
        schedule[step_ids, int(row.segment_id)] = np.minimum(
            schedule[step_ids, int(row.segment_id)],
            float(row.capacity_factor),
        )
    return torch.tensor(schedule, dtype=torch.float32)


def _km_distance_to_segment(km: float, km_start: float, km_end: float) -> float:
    if km < km_start:
        return float(km_start - km)
    if km > km_end:
        return float(km - km_end)
    return 0.0


def nearest_segment_candidates(
    unmapped: pd.DataFrame,
    metadata_csv: str,
    top_k: int = 2,
) -> pd.DataFrame:
    if unmapped.empty:
        return pd.DataFrame()

    meta = pd.read_csv(metadata_csv).copy()
    rows: list[dict[str, object]] = []

    for incident in unmapped.itertuples(index=False):
        br = int(incident.br)
        km = float(incident.km)
        candidates = meta[meta["br"] == br].copy()
        if candidates.empty:
            rows.append(
                {
                    "data_hora_inicial": incident.data_hora_inicial,
                    "data_hora_final": incident.data_hora_final,
                    "br": br,
                    "km": km,
                    "Amplitude": incident.Amplitude,
                    "candidate_rank": 1,
                    "segment_id": np.nan,
                    "direction": np.nan,
                    "km_start": np.nan,
                    "km_end": np.nan,
                    "km_distance": np.nan,
                    "reason": "br_not_in_metadata",
                }
            )
            continue

        candidates["km_distance"] = candidates.apply(
            lambda x: _km_distance_to_segment(km, float(x.km_start), float(x.km_end)),
            axis=1,
        )
        top = candidates.sort_values(["km_distance", "segment_id"]).head(top_k)
        for rank, seg in enumerate(top.itertuples(index=False), start=1):
            rows.append(
                {
                    "data_hora_inicial": incident.data_hora_inicial,
                    "data_hora_final": incident.data_hora_final,
                    "br": br,
                    "km": km,
                    "Amplitude": incident.Amplitude,
                    "candidate_rank": rank,
                    "segment_id": int(seg.segment_id),
                    "direction": str(seg.direction),
                    "km_start": float(seg.km_start),
                    "km_end": float(seg.km_end),
                    "km_distance": float(seg.km_distance),
                    "reason": "nearest_on_same_br",
                }
            )
    return pd.DataFrame(rows)


def write_incident_mapping_reports(
    mapping: IncidentMappingResult,
    metadata_csv: str,
    output_dir: str | Path,
    top_k: int = 2,
) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mapped_path = out_dir / "incident_mapped_segments.csv"
    unmapped_path = out_dir / "incident_unmapped.csv"
    nearest_path = out_dir / "incident_unmapped_nearest_segments.csv"
    summary_path = out_dir / "incident_mapping_summary.json"

    mapping.mapped.to_csv(mapped_path, index=False)
    mapping.unmapped.to_csv(unmapped_path, index=False)

    nearest = nearest_segment_candidates(mapping.unmapped, metadata_csv=metadata_csv, top_k=top_k)
    nearest.to_csv(nearest_path, index=False)

    summary = {
        "mapped_rows": mapping.mapped_rows,
        "unmapped_rows": mapping.unmapped_rows,
        "mapped_unique_incidents": int(
            mapping.mapped[["data_hora_inicial", "data_hora_final", "br", "km"]]
            .drop_duplicates()
            .shape[0]
        )
        if not mapping.mapped.empty
        else 0,
        "unmapped_unique_incidents": int(
            mapping.unmapped[["data_hora_inicial", "data_hora_final", "br", "km"]]
            .drop_duplicates()
            .shape[0]
        )
        if not mapping.unmapped.empty
        else 0,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "mapped_csv": str(mapped_path),
        "unmapped_csv": str(unmapped_path),
        "nearest_csv": str(nearest_path),
        "summary_json": str(summary_path),
    }
