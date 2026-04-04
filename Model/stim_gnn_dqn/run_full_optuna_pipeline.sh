#!/usr/bin/env bash
set -euo pipefail

cd "/home/andina/Tesis_storm_v2/Tesis Code Storm v2"
source .venv/bin/activate
export PYTHONPATH=Model

mkdir -p Model/outputs

echo "[1/3] Running Optuna (50 trials, full data)"
python Model/stim_gnn_dqn/tune_full_optuna.py | tee Model/outputs/optuna_full.log

echo "[2/3] Training final model with best params"
python - <<'PY'
import json
from dataclasses import replace
from pathlib import Path

from stim_gnn_dqn.config import Config
from stim_gnn_dqn.train import train_rl

best = json.loads(Path("Model/outputs/optuna_best_params_full.json").read_text(encoding="utf-8"))["best_params"]
cfg = replace(
    Config(),
    reward_alpha=0.01,
    reward_kappa=best["reward_kappa"],
    reward_beta=best["reward_beta"],
    reward_eta=best["reward_eta"],
    reward_deactivation_lambda=best["reward_deactivation_lambda"],
    gconv_hops_k=best["gconv_hops_k"],
    dqn_lookahead_h=best["dqn_lookahead_h"],
    max_budget=10,
    episodes=180,
    workdir="Model/outputs/final_optuna_full",
)
train_rl(cfg, save_artifacts=True)
print("DONE:", cfg.workdir)
PY

echo "[3/3] Evaluating best k (budget sweep)"
python - <<'PY'
import csv
import json
from dataclasses import replace
from pathlib import Path
from statistics import mean

import torch

from stim_gnn_dqn.config import Config
from stim_gnn_dqn.data_loader import build_dataset
from stim_gnn_dqn.environment import CapacityConstrainedEnv
from stim_gnn_dqn.evaluate import run_episode
from stim_gnn_dqn.model import GNNQNetwork
from stim_gnn_dqn.preprocess import build_static_parameters

best = json.loads(Path("Model/outputs/optuna_best_params_full.json").read_text(encoding="utf-8"))["best_params"]
cfg = replace(
    Config(),
    reward_alpha=0.01,
    reward_kappa=best["reward_kappa"],
    reward_beta=best["reward_beta"],
    reward_eta=best["reward_eta"],
    reward_deactivation_lambda=best["reward_deactivation_lambda"],
    gconv_hops_k=best["gconv_hops_k"],
    dqn_lookahead_h=best["dqn_lookahead_h"],
    workdir="Model/outputs/final_optuna_full",
)

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = build_dataset(**cfg.dataset_kwargs)
static = build_static_parameters(
    dataset=dataset,
    horizon=cfg.horizon,
    train_ratio=cfg.train_ratio,
    val_ratio=cfg.val_ratio,
    q=cfg.capacity_quantile,
    threshold_alpha=cfg.threshold_alpha,
    validation_start=cfg.validation_start,
    validation_end=cfg.validation_end,
)

env0 = CapacityConstrainedEnv(
    dataset=dataset,
    static=static,
    horizon=cfg.horizon,
    max_budget=10,
    history_len=cfg.history_len,
    reward_alpha=cfg.reward_alpha,
    reward_beta=cfg.reward_beta,
    reward_kappa=cfg.reward_kappa,
    reward_eta=cfg.reward_eta,
    reward_deactivation_lambda=cfg.reward_deactivation_lambda,
    reward_saturation_threshold=cfg.reward_saturation_threshold,
    reward_saturation_weight=cfg.reward_saturation_weight,
    reward_zeta=cfg.reward_zeta,
    incident_prob=cfg.incident_prob_eval,
    incident_factor_low=cfg.incident_factor_low,
    incident_factor_high=cfg.incident_factor_high,
    incident_duration_min=cfg.incident_duration_min,
    incident_duration_max=cfg.incident_duration_max,
    incident_edge_fraction=cfg.incident_edge_fraction,
    device=device,
)

s0 = env0.reset(static.val_starts[0])
q_net = GNNQNetwork(
    node_in_dim=s0.x.shape[1],
    edge_in_dim=s0.edge_attr.shape[1],
    hidden_dim=cfg.hidden_dim,
    gnn_layers=cfg.gnn_layers,
    spatial_hops=(cfg.gconv_hops_k if cfg.gconv_hops_k is not None else cfg.gnn_layers),
    global_temporal_dim=2,
).to(device)
q_net.load_state_dict(torch.load(Path(cfg.workdir) / "best_q_net.pt", map_location=device))
q_net.eval()

rows = []
starts = static.val_starts[:3]
for k in range(1, 13):
    env = CapacityConstrainedEnv(
        dataset=dataset,
        static=static,
        horizon=cfg.horizon,
        max_budget=k,
        history_len=cfg.history_len,
        reward_alpha=cfg.reward_alpha,
        reward_beta=cfg.reward_beta,
        reward_kappa=cfg.reward_kappa,
        reward_eta=cfg.reward_eta,
        reward_deactivation_lambda=cfg.reward_deactivation_lambda,
        reward_saturation_threshold=cfg.reward_saturation_threshold,
        reward_saturation_weight=cfg.reward_saturation_weight,
        reward_zeta=cfg.reward_zeta,
        incident_prob=cfg.incident_prob_eval,
        incident_factor_low=cfg.incident_factor_low,
        incident_factor_high=cfg.incident_factor_high,
        incident_duration_min=cfg.incident_duration_min,
        incident_duration_max=cfg.incident_duration_max,
        incident_edge_fraction=cfg.incident_edge_fraction,
        device=device,
    )
    ms = [run_episode(env, "rl", q_net, s) for s in starts]
    rows.append(
        {
            "k": k,
            "J": mean(m.cumulative_reward for m in ms),
            "J_int": mean(m.j_int for m in ms),
            "phi": mean(m.flicker_rate for m in ms),
            "MTTD": mean(m.mean_time_to_deactivation for m in ms),
            "avg_budget_used": mean(m.avg_budget_used for m in ms),
        }
    )

out = Path(cfg.workdir) / "best_k_eval.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

best_k = max(rows, key=lambda r: r["J_int"])
print("BEST_K_BY_J_INT:", best_k)
print("SAVED:", out)
PY

echo "Pipeline finished successfully."
