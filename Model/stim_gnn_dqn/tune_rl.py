from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import optuna
import torch

from .config import Config
from .data_loader import build_dataset
from .environment import CapacityConstrainedEnv
from .evaluate import run_episode
from .incidents import build_incident_schedule_for_episode, map_incidents_to_segments
from .preprocess import build_static_parameters
from .train import train_rl


def _build_eval_env(cfg: Config, dataset, static, device: str) -> CapacityConstrainedEnv:
    return CapacityConstrainedEnv(
        dataset=dataset,
        static=static,
        horizon=cfg.horizon,
        max_budget=cfg.max_budget,
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


def objective(trial: optuna.trial.Trial) -> float:
    # 1. Suggest hyperparameters (reward/penalty balance + optimization stability).
    alpha = trial.suggest_float("reward_alpha", 0.05, 0.40)
    eta = trial.suggest_float("reward_eta", 0.1, 1.5)
    beta = trial.suggest_float("reward_beta", 2.0, 8.0)
    kappa = trial.suggest_float("reward_kappa", 0.01, 0.05)
    lr = trial.suggest_float("lr", 5e-5, 2e-4, log=True)
    tau = trial.suggest_float("target_update_tau", 0.001, 0.01)

    # 2. Initialize config for short tuning runs.
    cfg = replace(
        Config(),
        reward_alpha=alpha,
        reward_eta=eta,
        reward_beta=beta,
        reward_kappa=kappa,
        lr=lr,
        target_update_tau=tau,
        epsilon_decay_steps=5000,
        episodes=60,
        max_budget=10,
    )

    # 3. Run mini-training.
    try:
        model, _history = train_rl(cfg, save_artifacts=False)
    except Exception as exc:
        print(f"Trial failed due to: {exc}")
        return -9999.0

    # 4. Evaluate on validation episodes (shared schedules when available).
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
    env = _build_eval_env(cfg, dataset, static, device=device)

    mapped_incidents = None
    if (
        dataset.timestamps is not None
        and cfg.incident_events_xlsx is not None
        and cfg.incident_metadata_csv is not None
        and Path(cfg.incident_events_xlsx).exists()
        and Path(cfg.incident_metadata_csv).exists()
    ):
        mapped_incidents = map_incidents_to_segments(
            incidents_xlsx=cfg.incident_events_xlsx,
            metadata_csv=cfg.incident_metadata_csv,
            both_factor=cfg.incident_both_directions_factor,
            single_factor=cfg.incident_single_direction_factor,
        ).mapped

    model.eval()
    val_j_int = []
    budgets = []
    for start_t in static.val_starts[:5]:
        incident_schedule = None
        if mapped_incidents is not None:
            incident_schedule = build_incident_schedule_for_episode(
                start_t=start_t,
                horizon=cfg.horizon,
                num_edges=dataset.num_edges,
                timestamps=dataset.timestamps,
                mapped_incidents=mapped_incidents,
            )
        metrics = run_episode(env, "rl", model, start_t, incident_schedule=incident_schedule)
        val_j_int.append(metrics.j_int)
        budgets.append(metrics.avg_budget_used)

    if not val_j_int:
        return -9999.0

    avg_j_int = sum(val_j_int) / len(val_j_int)
    avg_budget = sum(budgets) / len(budgets)
    trial.set_user_attr("avg_budget", float(avg_budget))

    # 5. Penalize "do nothing" policies.
    if avg_budget < 0.5:
        return -5000.0 + float(avg_j_int)
    return float(avg_j_int)


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("Best Trial:")
    print(study.best_params)

    out_path = Path(Config().workdir_path) / "best_hyperparams.json"
    out_payload = {
        "best_params": study.best_params,
        "best_value": study.best_value,
    }
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(f"Saved best hyperparameters to {out_path}")
