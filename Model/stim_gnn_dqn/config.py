from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # Preferred inputs (directly compatible with flow imputer outputs)
    network_json: str | None = field(
        default_factory=lambda: str(_project_root().parent / "Data" / "data" / "network.json")
    )
    imputed_flow_npz: str | None = field(
        default_factory=lambda: str(
            _project_root() / "outputs_imputation" / "imputed_edge_flow_matrix.npz"
        )
    )

    # Legacy inputs (kept for backward compatibility)
    graph_json: str | None = field(
        default_factory=lambda: str(_project_root() / "data" / "graph_dict_rodovia.json")
    )
    positions_xlsx: str | None = field(
        default_factory=lambda: str(_project_root() / "data" / "e2_posicoes_geohash.xlsx")
    )
    distance_xlsx: str | None = field(
        default_factory=lambda: str(_project_root() / "data" / "e2_rede_por_distancia.xlsx")
    )
    workdir: str = field(default_factory=lambda: str(_project_root() / "outputs"))

    # Temporal structure
    horizon: int = 24
    history_len: int = 4
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    validation_start: str | None = "2024-04-30 02:30:00"
    validation_end: str | None = "2024-06-22 19:14:00"

    # Stationary structural parameters
    capacity_quantile: float = 0.95
    threshold_alpha: float = 0.66

    # Best-performing single-stream SMA reward setup.
    reward_alpha: float = 0.01
    reward_beta: float = 5.00
    reward_kappa: float = 0.15
    reward_eta: float = 5.00
    reward_deactivation_lambda: float = 0.10
    reward_saturation_threshold: float = 0.72
    reward_saturation_weight: float = 5.00
    reward_zeta: float = 0.05
    gamma: float = 0.99

    # Action and model
    # max_budget is the per-snapshot intervention ceiling. The policy may
    # spend any amount from 0..max_budget at a given step.
    max_budget: int = 10
    hidden_dim: int = 64
    gnn_layers: int = 3

    # Optimization
    episodes: int = 180
    lr: float = 1e-4
    lr_min: float = 1e-5
    batch_size: int = 64
    replay_capacity: int = 5000
    warmup_steps: int = 64
    target_update_tau: float = 0.005
    max_grad_norm: float = 1.0
    seed: int = 7

    # Epsilon-greedy
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000

    # Validation model selection during training.
    val_eval_episodes: int = 4
    val_eval_interval: int = 1

    # Incident augmentation (PDF real-time recommendation)
    incident_prob_train: float = 0.35
    incident_prob_eval: float = 1.0
    incident_factor_low: float = 0.2
    incident_factor_high: float = 0.6
    incident_duration_min: int = 2
    incident_duration_max: int = 6
    incident_edge_fraction: float = 0.10
    incident_events_xlsx: str | None = field(
        default_factory=lambda: str(
            _project_root().parent / "Data" / "data" / "ocorrencias-interdicoes_rs_chuvas_2024.xlsx"
        )
    )
    incident_metadata_csv: str | None = field(
        default_factory=lambda: str(_project_root() / "outputs_imputation" / "imputed_edge_metadata.csv")
    )
    incident_both_directions_factor: float = 0.20
    incident_single_direction_factor: float = 0.50

    # Evaluation
    eval_split: str = "val"
    eval_episodes: int = 4
    # Directional model guardrail for RS imputed output.
    expected_directed_edges: int | None = 852
    # Reporting-only normalization to keep reward magnitudes comparable with
    # previous undirected (426-edge) visual baselines.
    report_reward_normalization_edges: int | None = 852

    @property
    def workdir_path(self) -> Path:
        path = Path(self.workdir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def dataset_kwargs(self) -> dict[str, str | None]:
        return {
            "graph_json": self.graph_json,
            "positions_xlsx": self.positions_xlsx,
            "distance_xlsx": self.distance_xlsx,
            "network_json": self.network_json,
            "imputed_flow_npz": self.imputed_flow_npz,
            "expected_num_edges": self.expected_directed_edges,
        }
