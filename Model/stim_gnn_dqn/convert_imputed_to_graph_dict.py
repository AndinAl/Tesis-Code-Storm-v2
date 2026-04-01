from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_network_json() -> Path:
    return _project_root().parent / "Data" / "data" / "network.json"


def _default_imputed_npz() -> Path:
    return _project_root() / "outputs_imputation" / "imputed_edge_flow_matrix.npz"


def _default_output_json() -> Path:
    return _project_root() / "data" / "graph_dict_rodovia.json"


def _validate_mapping(network_json: Path, sources: list[str], targets: list[str]) -> tuple[int, int]:
    with network_json.open("r", encoding="utf-8") as f:
        network = json.load(f)

    undirected_edges = {
        tuple(sorted((str(edge["node1"]), str(edge["node2"]))))
        for edge in network.get("edges", [])
    }
    matched = 0
    for src, dst in zip(sources, targets):
        if tuple(sorted((src, dst))) in undirected_edges:
            matched += 1
    return matched, len(sources)


def convert_imputed_to_graph_dict(
    network_json: Path,
    imputed_flow_npz: Path,
    output_json: Path,
    max_steps: int | None = None,
) -> dict[str, int]:
    npz = np.load(imputed_flow_npz, allow_pickle=True)
    required = {"flow", "source", "target", "timestamps"}
    missing = sorted(required - set(npz.files))
    if missing:
        raise KeyError(f"Missing required arrays in npz: {', '.join(missing)}")

    flow = np.asarray(npz["flow"], dtype=np.float32)
    if flow.ndim != 2:
        raise ValueError(f"Expected 2D flow array [time, edge], got {flow.shape}")

    sources = [str(x) for x in npz["source"].tolist()]
    targets = [str(x) for x in npz["target"].tolist()]
    timestamps = [str(x) for x in npz["timestamps"].tolist()]

    num_steps, num_edges = flow.shape
    if len(sources) != num_edges or len(targets) != num_edges:
        raise ValueError(
            f"source/target length mismatch: flow has {num_edges} edges, "
            f"source={len(sources)}, target={len(targets)}"
        )
    if len(timestamps) != num_steps:
        raise ValueError(
            f"timestamps length mismatch: flow has {num_steps} steps, "
            f"timestamps={len(timestamps)}"
        )

    steps = min(num_steps, max_steps) if max_steps is not None else num_steps
    edge_keys = [str((src, dst)) for src, dst in zip(sources, targets)]

    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as f:
        f.write("{")
        for t in range(steps):
            snapshot = {edge_keys[e]: float(flow[t, e]) for e in range(num_edges)}
            ts = timestamps[t]

            if t > 0:
                f.write(",")
            f.write("\n")
            f.write(json.dumps(ts, ensure_ascii=False))
            f.write(":")
            f.write(json.dumps(snapshot, ensure_ascii=False))

        if steps > 0:
            f.write("\n")
        f.write("}\n")

    matched, total = _validate_mapping(network_json, sources, targets)
    return {
        "written_steps": steps,
        "num_edges": num_edges,
        "matched_edges": matched,
        "total_directed_edges": total,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert imputed_edge_flow_matrix.npz into the legacy "
            "graph_dict_rodovia.json snapshot format."
        )
    )
    parser.add_argument(
        "--network-json",
        type=Path,
        default=_default_network_json(),
        help="Path to Data/data/network.json",
    )
    parser.add_argument(
        "--imputed-flow-npz",
        type=Path,
        default=_default_imputed_npz(),
        help="Path to outputs_imputation/imputed_edge_flow_matrix.npz",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=_default_output_json(),
        help="Path to write graph_dict_rodovia.json",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on the number of timestamps to export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = convert_imputed_to_graph_dict(
        network_json=args.network_json,
        imputed_flow_npz=args.imputed_flow_npz,
        output_json=args.output_json,
        max_steps=args.max_steps,
    )
    print(f"Wrote: {args.output_json}")
    print(
        "Exported steps={written_steps}, edges_per_step={num_edges}".format(
            written_steps=stats["written_steps"],
            num_edges=stats["num_edges"],
        )
    )
    print(
        "Directed-edge mapping matched against network.json: "
        "{matched_edges}/{total_directed_edges}".format(**stats)
    )


if __name__ == "__main__":
    main()
