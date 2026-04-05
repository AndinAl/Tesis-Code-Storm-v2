from __future__ import annotations

import json
import math
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

from .config import ImputationConfig


@dataclass(frozen=True)
class RoadNode:
    geohash: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class RoadEdge:
    edge_idx: int
    node1: str
    node2: str
    br: int
    km_start: float
    km_end: float
    length_km: float
    midpoint_lat: float
    midpoint_lon: float


@dataclass(frozen=True)
class DirectedSegment:
    segment_id: int
    edge_idx: int
    source: str
    target: str
    direction: str
    br: int
    km_start: float
    km_end: float
    length_km: float
    midpoint_lat: float
    midpoint_lon: float


@dataclass(frozen=True)
class SegmentRange:
    km_start: float
    km_end: float
    edge_idx: int


@dataclass
class FlowImputationDataset:
    directed_segments: List[DirectedSegment]
    timestamps: pd.DatetimeIndex
    edge_index: torch.LongTensor
    edge_weight: torch.FloatTensor
    observed_flow: torch.FloatTensor
    normalized_flow: torch.FloatTensor
    observed_mask: torch.BoolTensor
    static_features: torch.FloatTensor
    temporal_features: torch.FloatTensor
    flow_scale: float
    train_end: int
    val_end: int
    matched_sensor_pairs: int
    unmatched_sensor_pairs: List[tuple[int, float]]

    @property
    def num_steps(self) -> int:
        return int(self.observed_flow.shape[0])

    @property
    def num_segments(self) -> int:
        return len(self.directed_segments)

    @property
    def feature_dim(self) -> int:
        return int(2 + self.temporal_features.shape[1] + self.static_features.shape[1])

    def split_bounds(self, split: str) -> tuple[int, int]:
        if split == "train":
            return 0, self.train_end
        if split == "val":
            return self.train_end, self.val_end
        if split == "test":
            return self.val_end, self.num_steps
        raise ValueError(f"Unknown split: {split}")


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return default
    return float(text.replace(",", "."))


def _safe_zscore(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    std = float(arr.std())
    if std < 1e-6:
        return np.zeros_like(arr)
    return (arr - float(arr.mean())) / (std + 1e-6)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * radius_km * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1.0 - a)))


def load_rs_network(network_json: str) -> tuple[Dict[str, RoadNode], List[RoadEdge]]:
    raw = json.loads(Path(network_json).read_text(encoding="utf-8"))

    nodes: Dict[str, RoadNode] = {}
    for geohash, payload in raw["nodes"].items():
        lat_text, lon_text = [part.strip() for part in payload["ponto"].split(",")]
        nodes[geohash] = RoadNode(
            geohash=payload.get("geohash", geohash),
            latitude=float(lat_text),
            longitude=float(lon_text),
        )

    road_edges: List[RoadEdge] = []
    for edge_idx, raw_edge in enumerate(raw["edges"]):
        data = raw_edge["data"]
        node1 = raw_edge["node1"]
        node2 = raw_edge["node2"]
        point1 = nodes[node1]
        point2 = nodes[node2]
        km_start = _as_float(data.get("km_inicial"))
        km_end = _as_float(data.get("km_final"))
        length_km = _as_float(data.get("distancia"), default=max(km_end - km_start, 0.0))
        road_edges.append(
            RoadEdge(
                edge_idx=edge_idx,
                node1=node1,
                node2=node2,
                br=int(data["br"]),
                km_start=km_start,
                km_end=km_end,
                length_km=length_km,
                midpoint_lat=0.5 * (point1.latitude + point2.latitude),
                midpoint_lon=0.5 * (point1.longitude + point2.longitude),
            )
        )

    return nodes, road_edges


def build_directed_segments(road_edges: Sequence[RoadEdge]) -> List[DirectedSegment]:
    segments: List[DirectedSegment] = []
    for edge in road_edges:
        segments.append(
            DirectedSegment(
                segment_id=2 * edge.edge_idx,
                edge_idx=edge.edge_idx,
                source=edge.node1,
                target=edge.node2,
                direction="C",
                br=edge.br,
                km_start=edge.km_start,
                km_end=edge.km_end,
                length_km=edge.length_km,
                midpoint_lat=edge.midpoint_lat,
                midpoint_lon=edge.midpoint_lon,
            )
        )
        segments.append(
            DirectedSegment(
                segment_id=2 * edge.edge_idx + 1,
                edge_idx=edge.edge_idx,
                source=edge.node2,
                target=edge.node1,
                direction="D",
                br=edge.br,
                km_start=edge.km_start,
                km_end=edge.km_end,
                length_km=edge.length_km,
                midpoint_lat=edge.midpoint_lat,
                midpoint_lon=edge.midpoint_lon,
            )
        )
    return segments


def build_segment_line_graph(
    directed_segments: Sequence[DirectedSegment],
) -> tuple[torch.LongTensor, torch.FloatTensor]:
    segment_by_id = {segment.segment_id: segment for segment in directed_segments}
    endpoint_to_segments: dict[str, list[int]] = defaultdict(list)
    for segment in directed_segments:
        endpoint_to_segments[segment.source].append(segment.segment_id)
        endpoint_to_segments[segment.target].append(segment.segment_id)

    weighted_pairs: dict[tuple[int, int], float] = {}
    for segment in directed_segments:
        weighted_pairs[(segment.segment_id, segment.segment_id)] = 1.0

    for incident in endpoint_to_segments.values():
        for left in incident:
            for right in incident:
                if left == right:
                    continue
                left_seg = segment_by_id[left]
                right_seg = segment_by_id[right]
                dist_km = _haversine_km(
                    left_seg.midpoint_lat,
                    left_seg.midpoint_lon,
                    right_seg.midpoint_lat,
                    right_seg.midpoint_lon,
                )
                pair = (left, right)
                weighted_pairs[pair] = max(weighted_pairs.get(pair, 0.0), 1.0 / (1.0 + dist_km))

    row = [pair[0] for pair in weighted_pairs]
    col = [pair[1] for pair in weighted_pairs]
    weight = [weighted_pairs[pair] for pair in weighted_pairs]
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_weight = torch.tensor(weight, dtype=torch.float32)
    return edge_index, edge_weight


def load_hourly_traffic_table(
    traffic_parquet: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    df = pd.read_parquet(traffic_parquet)
    flow_columns = [
        "onibus_cam_2eixos",
        "onibus_cam_3eixos",
        "cam_4eixos",
        "cam_5eixos",
        "cam_6eixos",
        "cam_7eixos",
        "cam_8eixos",
        "cam_9eixos",
        "passeio",
        "motocicleta",
        "indefinido",
    ]

    for column in flow_columns:
        df[column] = (
            pd.to_numeric(
                df[column].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
            .fillna(0.0)
            .astype(np.int32)
        )

    df = df.dropna().copy()
    df["Total"] = df[flow_columns].sum(axis=1).astype(np.float32)
    df["Data"] = pd.to_datetime(df["Data"], dayfirst=True, errors="coerce")
    df["Hora"] = pd.to_numeric(df["Hora"], errors="coerce")
    df["BR"] = pd.to_numeric(df["BR"], errors="coerce")
    df["Km"] = pd.to_numeric(
        df["Km"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    df["Sentido"] = df["Sentido"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["Data", "Hora", "BR", "Km"]).copy()
    df["Hora"] = df["Hora"].astype(np.int16)
    df["BR"] = df["BR"].astype(np.int16)
    df["timestamp"] = df["Data"].dt.normalize() + pd.to_timedelta(df["Hora"], unit="h")

    if start_date is not None:
        df = df[df["timestamp"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        inclusive_end = pd.Timestamp(end_date) + pd.Timedelta(hours=23)
        df = df[df["timestamp"] <= inclusive_end]

    return df[["timestamp", "BR", "Km", "Sentido", "Total"]].reset_index(drop=True)


def _find_segment(segment_ranges: Sequence[SegmentRange], km_value: float) -> SegmentRange | None:
    km_starts = [segment.km_start for segment in segment_ranges]
    idx = bisect_right(km_starts, km_value) - 1
    if idx >= 0 and segment_ranges[idx].km_start <= km_value <= segment_ranges[idx].km_end:
        return segment_ranges[idx]
    return None


def _build_sensor_mapping(
    road_edges: Sequence[RoadEdge],
    hourly_counts: pd.DataFrame,
) -> tuple[pd.DataFrame, List[tuple[int, float]]]:
    ranges_by_br: dict[int, list[SegmentRange]] = defaultdict(list)
    for edge in road_edges:
        ranges_by_br[edge.br].append(
            SegmentRange(km_start=edge.km_start, km_end=edge.km_end, edge_idx=edge.edge_idx)
        )
    for ranges in ranges_by_br.values():
        ranges.sort(key=lambda item: item.km_start)

    mapping_rows = []
    unmatched: List[tuple[int, float]] = []
    for br, km_value in hourly_counts[["BR", "Km"]].drop_duplicates().itertuples(index=False):
        segment = _find_segment(ranges_by_br[int(br)], float(km_value))
        if segment is None:
            unmatched.append((int(br), float(km_value)))
            continue
        mapping_rows.append({"BR": int(br), "Km": float(km_value), "edge_idx": int(segment.edge_idx)})

    return pd.DataFrame(mapping_rows), unmatched


def _build_dense_flow_matrix(
    timestamps: pd.DatetimeIndex,
    num_segments: int,
    grouped_flow: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    dense_flow = np.zeros((len(timestamps), num_segments), dtype=np.float32)
    observed_mask = np.zeros((len(timestamps), num_segments), dtype=bool)

    time_indices = timestamps.get_indexer(grouped_flow["timestamp"])
    valid = time_indices >= 0
    segment_indices = grouped_flow.loc[valid, "segment_id"].to_numpy(dtype=np.int64)
    values = grouped_flow.loc[valid, "Total"].to_numpy(dtype=np.float32)

    dense_flow[time_indices[valid], segment_indices] = values
    observed_mask[time_indices[valid], segment_indices] = True
    return dense_flow, observed_mask


def prepare_imputation_dataset(cfg: ImputationConfig | None = None) -> FlowImputationDataset:
    cfg = cfg or ImputationConfig()
    _, road_edges = load_rs_network(cfg.network_json)
    directed_segments = build_directed_segments(road_edges)
    edge_index, edge_weight = build_segment_line_graph(directed_segments)

    hourly_counts = load_hourly_traffic_table(
        cfg.traffic_parquet,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
    )
    sensor_mapping, unmatched_sensor_pairs = _build_sensor_mapping(road_edges, hourly_counts)
    matched_counts = hourly_counts.merge(sensor_mapping, on=["BR", "Km"], how="inner")
    matched_counts = matched_counts[matched_counts["Sentido"].isin(["C", "D"])].copy()
    matched_counts["edge_idx"] = matched_counts["edge_idx"].astype(np.int64)
    matched_counts["segment_id"] = matched_counts["edge_idx"] * 2 + (
        matched_counts["Sentido"] == "D"
    ).astype(np.int64)

    grouped_flow = (
        matched_counts.groupby(["timestamp", "segment_id"], as_index=False)["Total"]
        .sum()
        .sort_values(["timestamp", "segment_id"])
        .reset_index(drop=True)
    )

    start_ts = pd.Timestamp(cfg.start_date)
    end_ts = pd.Timestamp(cfg.end_date) + pd.Timedelta(hours=23)
    timestamps = pd.date_range(start_ts, end_ts, freq="h")
    dense_flow, observed_mask = _build_dense_flow_matrix(
        timestamps=timestamps,
        num_segments=len(directed_segments),
        grouped_flow=grouped_flow,
    )

    num_steps = len(timestamps)
    train_end = max(1, int(num_steps * cfg.train_ratio))
    val_end = max(train_end + 1, int(num_steps * (cfg.train_ratio + cfg.val_ratio)))
    val_end = min(val_end, num_steps - 1)

    log_flow = np.log1p(dense_flow)
    train_observed = log_flow[:train_end][observed_mask[:train_end]]
    flow_scale = float(np.quantile(train_observed, 0.95)) if train_observed.size else 1.0
    flow_scale = max(flow_scale, 1.0)
    normalized_flow = log_flow / flow_scale

    coverage = observed_mask[:train_end].mean(axis=0).astype(np.float32)
    observed_sum = normalized_flow[:train_end].sum(axis=0)
    observed_count = np.maximum(observed_mask[:train_end].sum(axis=0), 1)
    mean_train_flow = (observed_sum / observed_count).astype(np.float32)

    length_norm = _safe_zscore([segment.length_km for segment in directed_segments])
    lat_norm = _safe_zscore([segment.midpoint_lat for segment in directed_segments])
    lon_norm = _safe_zscore([segment.midpoint_lon for segment in directed_segments])
    direction_c = np.array(
        [1.0 if segment.direction == "C" else 0.0 for segment in directed_segments],
        dtype=np.float32,
    )
    direction_d = 1.0 - direction_c
    static_features = np.stack(
        [
            length_norm,
            lat_norm,
            lon_norm,
            direction_c,
            direction_d,
            coverage,
            mean_train_flow,
        ],
        axis=1,
    ).astype(np.float32)

    hour = timestamps.hour.to_numpy(dtype=np.float32)
    day_of_week = timestamps.dayofweek.to_numpy(dtype=np.float32)
    month = timestamps.month.to_numpy(dtype=np.float32) - 1.0
    temporal_features = np.stack(
        [
            np.sin(2.0 * np.pi * hour / 24.0),
            np.cos(2.0 * np.pi * hour / 24.0),
            np.sin(2.0 * np.pi * day_of_week / 7.0),
            np.cos(2.0 * np.pi * day_of_week / 7.0),
            np.sin(2.0 * np.pi * month / 12.0),
            np.cos(2.0 * np.pi * month / 12.0),
        ],
        axis=1,
    ).astype(np.float32)

    return FlowImputationDataset(
        directed_segments=list(directed_segments),
        timestamps=timestamps,
        edge_index=edge_index,
        edge_weight=edge_weight,
        observed_flow=torch.tensor(dense_flow, dtype=torch.float32),
        normalized_flow=torch.tensor(normalized_flow, dtype=torch.float32),
        observed_mask=torch.tensor(observed_mask, dtype=torch.bool),
        static_features=torch.tensor(static_features, dtype=torch.float32),
        temporal_features=torch.tensor(temporal_features, dtype=torch.float32),
        flow_scale=flow_scale,
        train_end=train_end,
        val_end=val_end,
        matched_sensor_pairs=int(sensor_mapping.shape[0]),
        unmatched_sensor_pairs=unmatched_sensor_pairs,
    )
