from __future__ import annotations

import copy
import csv
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from .config import Config
from .data_loader import build_dataset
from .environment import CapacityConstrainedEnv
from .evaluate import run_episode
from .model import GNNQNetwork, select_action
from .preprocess import build_static_parameters
from .replay_buffer import ReplayBuffer, Transition
from .train import build_optimizer, epsilon_by_step, optimize_model, set_seed, soft_update_target


SWEEP_GRID = [
    {"reward_eta": 0.80, "reward_kappa": 0.15},
    {"reward_eta": 0.95, "reward_kappa": 0.10},
    {"reward_eta": 1.10, "reward_kappa": 0.08},
    {"reward_eta": 1.20, "reward_kappa": 0.05},
]

SWEEP_EPISODES = 20
OPTUNA_EPISODES = 20

# Lazily initialized context reused across Optuna trials.
_OPTUNA_CONTEXT: dict[str, Any] = {}


def _train_model(
    cfg: Config,
    dataset,
    static,
    device: str,
) -> GNNQNetwork:
    env = CapacityConstrainedEnv(
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
        incident_prob=cfg.incident_prob_train,
        incident_factor_low=cfg.incident_factor_low,
        incident_factor_high=cfg.incident_factor_high,
        incident_duration_min=cfg.incident_duration_min,
        incident_duration_max=cfg.incident_duration_max,
        incident_edge_fraction=cfg.incident_edge_fraction,
        device=device,
    )

    sample_state = env.reset(static.train_starts[0])
    q_net = GNNQNetwork(
        node_in_dim=sample_state.x.shape[1],
        edge_in_dim=sample_state.edge_attr.shape[1],
        hidden_dim=cfg.hidden_dim,
        gnn_layers=cfg.gnn_layers,
        global_context_dim=(
            int(sample_state.global_context.numel()) if sample_state.global_context is not None else 0
        ),
    ).to(device)
    target_net = copy.deepcopy(q_net)
    optimizer = build_optimizer(q_net, cfg)
    buffer = ReplayBuffer(cfg.replay_capacity)

    global_step = 0
    for _ in range(cfg.episodes):
        start_t = static.train_starts[global_step % len(static.train_starts)]
        state = env.reset(start_t)
        done = False

        while not done:
            epsilon = epsilon_by_step(global_step, cfg)
            action_idx = select_action(q_net, state, cfg.max_budget, epsilon, device=device)
            next_state, out = env.step(action_idx.detach().cpu().tolist())

            buffer.push(
                Transition(
                    state=state,
                    action_idx=action_idx.detach().cpu(),
                    reward=out.reward,
                    next_state=next_state,
                    done=out.done,
                )
            )

            if len(buffer) >= cfg.warmup_steps:
                optimize_model(q_net, target_net, buffer, optimizer, cfg, device)

            soft_update_target(target_net, q_net, cfg.target_update_tau)

            state = next_state
            done = out.done
            global_step += 1

    return q_net


def _evaluate_model(cfg: Config, dataset, static, q_net: GNNQNetwork, device: str) -> dict[str, dict[str, float]]:
    starts = static.val_starts if cfg.eval_split == "val" else static.test_starts
    env = CapacityConstrainedEnv(
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

    policies = ["random", "greedy", "degree", "rl", "storm"]
    results = {name: [] for name in policies}

    for start_t in starts[: cfg.eval_episodes]:
        for name in policies:
            results[name].append(run_episode(env, name, q_net, start_t))

    summary = {}
    for name, items in results.items():
        summary[name] = {
            "J": mean(m.cumulative_reward for m in items),
            "J_int": mean(m.j_int for m in items),
            "MTTD": mean(m.mean_time_to_deactivation for m in items),
            "S": mean(m.stability_score for m in items),
            "phi": mean(m.flicker_rate for m in items),
            "D": mean(m.avg_deactivation_count for m in items),
            "B": mean(m.avg_budget_used for m in items),
        }
    return summary


def _ensure_optuna_context() -> dict[str, Any]:
    global _OPTUNA_CONTEXT
    if _OPTUNA_CONTEXT:
        return _OPTUNA_CONTEXT

    base_cfg = Config(episodes=OPTUNA_EPISODES)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = build_dataset(**base_cfg.dataset_kwargs)
    static = build_static_parameters(
        dataset=dataset,
        horizon=base_cfg.horizon,
        train_ratio=base_cfg.train_ratio,
        val_ratio=base_cfg.val_ratio,
        q=base_cfg.capacity_quantile,
        threshold_alpha=base_cfg.threshold_alpha,
        validation_start=base_cfg.validation_start,
        validation_end=base_cfg.validation_end,
    )
    _OPTUNA_CONTEXT = {
        "base_cfg": base_cfg,
        "dataset": dataset,
        "static": static,
        "device": device,
    }
    return _OPTUNA_CONTEXT


def objective(trial) -> float:
    """
    Optuna objective requested by user:
    - tune epsilon_decay_steps in [3000, 7000]
    - tune reward_kappa in [0.01, 0.10]
    """
    ctx = _ensure_optuna_context()
    base_cfg: Config = ctx["base_cfg"]
    dataset = ctx["dataset"]
    static = ctx["static"]
    device: str = ctx["device"]

    epsilon_decay = trial.suggest_int("epsilon_decay_steps", 3000, 7000)
    kappa = trial.suggest_float("reward_kappa", 0.01, 0.10)

    cfg = replace(
        base_cfg,
        reward_kappa=kappa,
        epsilon_decay_steps=epsilon_decay,
    )
    set_seed(cfg.seed + int(getattr(trial, "number", 0)))

    q_net = _train_model(cfg, dataset, static, device)
    summary = _evaluate_model(cfg, dataset, static, q_net, device)
    rl = summary["rl"]

    # Maximize validation-style cumulative reward of RL policy.
    trial.set_user_attr("rl_J_int", rl["J_int"])
    trial.set_user_attr("rl_MTTD", rl["MTTD"])
    trial.set_user_attr("rl_D", rl["D"])
    return float(rl["J"])


def run_optuna(n_trials: int = 20):
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("optuna is not installed in this environment.") from exc

    study = optuna.create_study(direction="maximize", study_name="stim_gnn_dqn_exploration_kappa")
    study.optimize(objective, n_trials=n_trials)
    return study


def main() -> None:
    base_cfg = Config(episodes=SWEEP_EPISODES)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = build_dataset(**base_cfg.dataset_kwargs)
    static = build_static_parameters(
        dataset=dataset,
        horizon=base_cfg.horizon,
        train_ratio=base_cfg.train_ratio,
        val_ratio=base_cfg.val_ratio,
        q=base_cfg.capacity_quantile,
        threshold_alpha=base_cfg.threshold_alpha,
        validation_start=base_cfg.validation_start,
        validation_end=base_cfg.validation_end,
    )

    out_path = Path(base_cfg.workdir_path) / "reward_sweep.csv"
    rows = []

    for combo in SWEEP_GRID:
        cfg = replace(base_cfg, **combo)
        set_seed(cfg.seed)

        print(
            f"\n=== Sweep eta={cfg.reward_eta:.2f}, "
            f"kappa={cfg.reward_kappa:.2f}, episodes={cfg.episodes} ==="
        )
        q_net = _train_model(cfg, dataset, static, device)
        summary = _evaluate_model(cfg, dataset, static, q_net, device)
        rl = summary["rl"]
        greedy = summary["greedy"]
        degree = summary["degree"]

        row = {
            "reward_eta": cfg.reward_eta,
            "reward_kappa": cfg.reward_kappa,
            "train_episodes": cfg.episodes,
            "rl_J": rl["J"],
            "rl_J_int": rl["J_int"],
            "rl_MTTD": rl["MTTD"],
            "rl_D": rl["D"],
            "rl_B": rl["B"],
            "greedy_J_int": greedy["J_int"],
            "greedy_MTTD": greedy["MTTD"],
            "degree_MTTD": degree["MTTD"],
        }
        rows.append(row)
        print(
            f"rl | J={rl['J']:.3f} J_int={rl['J_int']:.3f} "
            f"MTTD={rl['MTTD']:.3f} D={rl['D']:.3f} B={rl['B']:.3f}"
        )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved sweep results to {out_path}")


if __name__ == "__main__":
    main()
