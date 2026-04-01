from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch


@dataclass
class STIMDataset:
    node_names: List[str]
    edge_names: List[Tuple[str, str]]
    node_to_idx: Dict[str, int]
    edge_index: torch.LongTensor
    flows: torch.FloatTensor
    distances: torch.FloatTensor
    coords: torch.FloatTensor
    community_ids: torch.LongTensor
    gateway_scores: torch.FloatTensor
    indegree: torch.FloatTensor
    outdegree: torch.FloatTensor
    edge_directions: List[str] | None = None
    timestamps: np.ndarray | None = None

    @property
    def betweenness_centrality(self) -> torch.FloatTensor:
        # Historically named gateway_scores in this repo, but the value is
        # exact node betweenness centrality from NetworkX.
        return self.gateway_scores

    @property
    def num_nodes(self) -> int:
        return len(self.node_names)

    @property
    def num_edges(self) -> int:
        return len(self.edge_names)

    @property
    def num_steps(self) -> int:
        return int(self.flows.shape[0])


def _parse_edge_key(key: str) -> Tuple[str, str]:
    src, dst = ast.literal_eval(key)
    return str(src), str(dst)


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return default
    return float(text.replace(",", "."))


def _parse_point(ponto: object) -> tuple[float, float]:
    coords = re.findall(r"[-+]?\d*\.?\d+", str(ponto))
    if len(coords) < 2:
        return 0.0, 0.0
    return float(coords[0]), float(coords[1])


def _compute_communities_and_gateways(node_names: List[str], edge_names: List[Tuple[str, str]]):
    g = nx.Graph()
    g.add_nodes_from(node_names)
    g.add_edges_from(edge_names)

    try:
        communities = list(nx.algorithms.community.greedy_modularity_communities(g))
    except Exception:
        communities = [set(node_names)]

    node_to_comm = {node: 0 for node in node_names}
    for cid, group in enumerate(communities):
        for node in group:
            node_to_comm[node] = cid

    bt = nx.betweenness_centrality(g, normalized=True)
    gateway_scores = np.array([bt.get(node, 0.0) for node in node_names], dtype=np.float32)
    community_ids = np.array([node_to_comm[node] for node in node_names], dtype=np.int64)
    return community_ids, gateway_scores


def _build_dataset_from_graph_dict(
    graph_json: str,
    positions_xlsx: str,
    distance_xlsx: str,
) -> STIMDataset:
    import pandas as pd

    graph_path = Path(graph_json)
    pos_path = Path(positions_xlsx)
    dist_path = Path(distance_xlsx)

    with graph_path.open("r", encoding="utf-8") as f:
        graph_dict = json.load(f)

    keys = list(graph_dict.keys())
    first_snapshot = graph_dict[keys[0]]
    edge_names = [_parse_edge_key(k) for k in first_snapshot.keys()]

    node_names = sorted({u for u, _ in edge_names} | {v for _, v in edge_names})
    node_to_idx = {node: i for i, node in enumerate(node_names)}

    edge_index_np = np.array(
        [[node_to_idx[u] for u, v in edge_names], [node_to_idx[v] for u, v in edge_names]],
        dtype=np.int64,
    )
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    flows = np.zeros((len(keys), len(edge_names)), dtype=np.float32)
    for t, key in enumerate(keys):
        snapshot = graph_dict[key]
        for e_idx, edge_key in enumerate(first_snapshot.keys()):
            flows[t, e_idx] = float(snapshot.get(edge_key, 0.0))

    pos_df = pd.read_excel(pos_path)
    pos_lookup = {
        str(row["geohash"]): (float(row["latitude"]), float(row["longitude"]))
        for _, row in pos_df.iterrows()
    }
    coords = np.zeros((len(node_names), 2), dtype=np.float32)
    for i, node in enumerate(node_names):
        lat, lon = pos_lookup.get(node, (0.0, 0.0))
        coords[i] = [lat, lon]

    dist_df = pd.read_excel(dist_path)
    row_names = dist_df.iloc[:, 0].astype(str).tolist()
    col_names = [str(c) for c in dist_df.columns[1:]]
    values = dist_df.iloc[:, 1:].to_numpy(dtype=np.float32)
    distance_lookup = {
        (row_names[i], col_names[j]): float(values[i, j])
        for i in range(len(row_names))
        for j in range(len(col_names))
    }
    distances = np.array(
        [distance_lookup.get((u, v), 0.0) for u, v in edge_names],
        dtype=np.float32,
    )

    community_ids, gateway_scores = _compute_communities_and_gateways(node_names, edge_names)

    indegree = np.zeros(len(node_names), dtype=np.float32)
    outdegree = np.zeros(len(node_names), dtype=np.float32)
    for u, v in edge_names:
        outdegree[node_to_idx[u]] += 1.0
        indegree[node_to_idx[v]] += 1.0

    timestamps: np.ndarray | None
    try:
        timestamps = np.array(keys, dtype="datetime64[ns]")
    except Exception:
        timestamps = None

    return STIMDataset(
        node_names=node_names,
        edge_names=edge_names,
        node_to_idx=node_to_idx,
        edge_index=edge_index,
        flows=torch.tensor(flows, dtype=torch.float32),
        distances=torch.tensor(distances, dtype=torch.float32),
        coords=torch.tensor(coords, dtype=torch.float32),
        community_ids=torch.tensor(community_ids, dtype=torch.long),
        gateway_scores=torch.tensor(gateway_scores, dtype=torch.float32),
        indegree=torch.tensor(indegree, dtype=torch.float32),
        outdegree=torch.tensor(outdegree, dtype=torch.float32),
        timestamps=timestamps,
    )


def _build_dataset_from_imputed_npz(
    network_json: str,
    imputed_flow_npz: str,
    expected_num_edges: int | None = None,
) -> STIMDataset:
    network_path = Path(network_json)
    npz_path = Path(imputed_flow_npz)

    with network_path.open("r", encoding="utf-8") as f:
        network = json.load(f)

    npz = np.load(npz_path, allow_pickle=True)
    # Directional edge mode requires per-segment direction labels to avoid
    # collapsing both carriageways into an averaged undirected stream.
    required_npz_keys = {"flow", "source", "target", "direction"}
    missing = sorted(required_npz_keys - set(npz.files))
    if missing:
        raise KeyError(
            "imputed flow file is missing required arrays: "
            f"{', '.join(missing)}"
        )

    flows = np.asarray(npz["flow"], dtype=np.float32)
    sources = [str(x) for x in npz["source"].tolist()]
    targets = [str(x) for x in npz["target"].tolist()]
    directions = [str(x) for x in npz["direction"].tolist()]

    if flows.ndim != 2:
        raise ValueError(f"Expected 2D flow array [time, edge], got shape {flows.shape}")

    if flows.shape[1] != len(sources) or flows.shape[1] != len(targets) or flows.shape[1] != len(directions):
        raise ValueError(
            "Flow columns do not match source/target lengths: "
            f"flow_cols={flows.shape[1]} source={len(sources)} target={len(targets)} direction={len(directions)}"
        )

    if expected_num_edges is not None and flows.shape[1] != int(expected_num_edges):
        raise ValueError(
            "Directional imputed matrix has unexpected number of edges: "
            f"expected={expected_num_edges} got={flows.shape[1]}. "
            "This usually means the file is not the full directional export."
        )

    timestamps: np.ndarray | None = None
    if "timestamps" in npz.files:
        try:
            timestamps = np.asarray(npz["timestamps"]).astype("datetime64[ns]")
        except Exception:
            timestamps = None
        if timestamps is not None and len(timestamps) != flows.shape[0]:
            raise ValueError(
                "timestamps length does not match flow time dimension: "
                f"timestamps={len(timestamps)} flow_steps={flows.shape[0]}"
            )

    edge_names = list(zip(sources, targets))
    node_names = sorted({u for u, _ in edge_names} | {v for _, v in edge_names})
    node_to_idx = {node: i for i, node in enumerate(node_names)}

    edge_index_np = np.array(
        [[node_to_idx[u] for u, v in edge_names], [node_to_idx[v] for u, v in edge_names]],
        dtype=np.int64,
    )
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    coords_lookup: Dict[str, tuple[float, float]] = {}
    for geohash, payload in network.get("nodes", {}).items():
        node_id = str(payload.get("geohash", geohash))
        coords_lookup[node_id] = _parse_point(payload.get("ponto", ""))

    coords = np.zeros((len(node_names), 2), dtype=np.float32)
    for i, node in enumerate(node_names):
        lat, lon = coords_lookup.get(node, (0.0, 0.0))
        coords[i] = [lat, lon]

    # Map each undirected physical edge to its static distance.
    undirected_distance: Dict[tuple[str, str], List[float]] = {}
    for raw_edge in network.get("edges", []):
        u = str(raw_edge["node1"])
        v = str(raw_edge["node2"])
        data = raw_edge.get("data", {})
        dist = _as_float(data.get("distancia"), default=0.0)
        if dist <= 0.0:
            km_i = _as_float(data.get("km_inicial"), default=0.0)
            km_f = _as_float(data.get("km_final"), default=0.0)
            dist = max(0.0, km_f - km_i)
        key = tuple(sorted((u, v)))
        undirected_distance.setdefault(key, []).append(dist)

    distance_lookup = {
        key: float(np.mean(vals))
        for key, vals in undirected_distance.items()
        if vals
    }
    has_npz_km = "km_start" in npz.files and "km_end" in npz.files
    km_start = np.asarray(npz["km_start"], dtype=np.float32) if has_npz_km else None
    km_end = np.asarray(npz["km_end"], dtype=np.float32) if has_npz_km else None

    distances = np.zeros(len(edge_names), dtype=np.float32)
    for i, (u, v) in enumerate(edge_names):
        pair = tuple(sorted((u, v)))
        if pair in distance_lookup:
            distances[i] = distance_lookup[pair]
        elif has_npz_km:
            distances[i] = float(abs(km_end[i] - km_start[i]))
        else:
            distances[i] = 0.0

    community_ids, gateway_scores = _compute_communities_and_gateways(node_names, edge_names)

    indegree = np.zeros(len(node_names), dtype=np.float32)
    outdegree = np.zeros(len(node_names), dtype=np.float32)
    for u, v in edge_names:
        outdegree[node_to_idx[u]] += 1.0
        indegree[node_to_idx[v]] += 1.0

    return STIMDataset(
        node_names=node_names,
        edge_names=edge_names,
        node_to_idx=node_to_idx,
        edge_index=edge_index,
        flows=torch.tensor(flows, dtype=torch.float32),
        distances=torch.tensor(distances, dtype=torch.float32),
        coords=torch.tensor(coords, dtype=torch.float32),
        community_ids=torch.tensor(community_ids, dtype=torch.long),
        gateway_scores=torch.tensor(gateway_scores, dtype=torch.float32),
        indegree=torch.tensor(indegree, dtype=torch.float32),
        outdegree=torch.tensor(outdegree, dtype=torch.float32),
        edge_directions=[str(x).upper() for x in directions],
        timestamps=timestamps,
    )


def build_dataset(
    graph_json: str | None = None,
    positions_xlsx: str | None = None,
    distance_xlsx: str | None = None,
    network_json: str | None = None,
    imputed_flow_npz: str | None = None,
    expected_num_edges: int | None = None,
) -> STIMDataset:
    # Preferred path: static RS network + imputed hourly edge flows.
    if network_json is not None or imputed_flow_npz is not None:
        if network_json is None or imputed_flow_npz is None:
            raise ValueError(
                "Both network_json and imputed_flow_npz must be provided "
                "when using imputed time-series input."
            )
        network_exists = Path(network_json).exists()
        imputed_exists = Path(imputed_flow_npz).exists()
        if network_exists and imputed_exists:
            return _build_dataset_from_imputed_npz(
                network_json,
                imputed_flow_npz,
                expected_num_edges=expected_num_edges,
            )

        # Backward-compatible fallback for environments still using the legacy
        # graph_dict + xlsx inputs.
        if (
            graph_json is not None
            and positions_xlsx is not None
            and distance_xlsx is not None
            and Path(graph_json).exists()
            and Path(positions_xlsx).exists()
            and Path(distance_xlsx).exists()
        ):
            return _build_dataset_from_graph_dict(graph_json, positions_xlsx, distance_xlsx)

        missing_paths: list[str] = []
        if not network_exists:
            missing_paths.append(str(network_json))
        if not imputed_exists:
            missing_paths.append(str(imputed_flow_npz))
        raise FileNotFoundError(
            "Could not load dataset from imputed input; missing file(s): "
            + ", ".join(missing_paths)
        )

    # Legacy fallback path: graph_dict_rodovia.json + xlsx assets.
    if graph_json is None or positions_xlsx is None or distance_xlsx is None:
        raise ValueError(
            "Either provide network_json+imputed_flow_npz, or provide "
            "graph_json+positions_xlsx+distance_xlsx."
        )
    return _build_dataset_from_graph_dict(graph_json, positions_xlsx, distance_xlsx)
