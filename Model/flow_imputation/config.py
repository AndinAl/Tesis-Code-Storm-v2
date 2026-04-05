from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class ImputationConfig:
    network_json: str = field(
        default_factory=lambda: str(_repo_root() / "Data" / "data" / "network.json")
    )
    traffic_parquet: str = field(
        default_factory=lambda: str(_repo_root() / "Data" / "data" / "pntc_rs_unificado.parquet")
    )
    workdir: str = field(
        default_factory=lambda: str(_repo_root() / "Model" / "outputs_imputation")
    )

    # Temporal bounds
    start_date: str = "2021-06-01"
    end_date: str = "2025-10-31"
    train_ratio: float = 0.70
    val_ratio: float = 0.15

    # Model architecture
    hidden_dim: int = 128
    cheb_k: int = 3
    dropout: float = 0.15

    # Training dynamics
    epochs: int = 50
    window_size: int = 168
    windows_per_epoch: int = 256
    eval_windows: int = 24
    holdout_ratio: float = 0.25
    min_window_observations: int = 24

    # Optimizer
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 0.5
    seed: int = 7

    model_filename: str = "stgnn_imputer.pt"
    history_filename: str = "stgnn_imputer_history.csv"
    completed_flow_filename: str = "imputed_edge_flow_matrix.npz"
    metadata_filename: str = "imputed_edge_metadata.csv"

    @property
    def workdir_path(self) -> Path:
        path = Path(self.workdir)
        path.mkdir(parents=True, exist_ok=True)
        return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
