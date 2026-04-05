from __future__ import annotations

import random
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from .baselines import (
    greedy_coverage_policy,
    lookahead_greedy_policy,
    random_policy,
    static_degree_policy,
)
from .config import Config
from .data_loader import build_dataset
from .environment import CapacityConstrainedEnv
from .incidents import (
    build_incident_schedule_for_episode,
    map_incidents_to_segments,
    write_incident_mapping_reports,
)
from .metrics import summarize_episode
from .model import GNNQNetwork, select_action
from .preprocess import build_static_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_storm_schedule(env: CapacityConstrainedEnv, factor: float = 0.6) -> torch.FloatTensor:
    schedule = torch.ones((env.horizon, env.dataset.num_edges), dtype=torch.float32)
    duration = min(6, env.horizon // 3)
    start = env.horizon // 2
    edge_count = max(1, int(env.dataset.num_edges * env.incident_edge_fraction))
    top_edges = torch.topk(env.static.capacity, k=min(edge_count, env.dataset.num_edges)).indices
    schedule[start : start + duration, top_edges] = factor
    return schedule


def run_episode(
    env: CapacityConstrainedEnv,
    policy_name: str,
    q_net: GNNQNetwork | None,
    start_t: int,
    incident_schedule: torch.FloatTensor | None = None,
    lookahead_h: int = 6,
    lookahead_gamma: float = 0.99,
):
    if policy_name == "storm":
        state = env.reset(start_t, incident_schedule=make_storm_schedule(env, factor=0.6))
        policy_name = "rl"
    else:
        state = env.reset(start_t, incident_schedule=incident_schedule)

    rewards, counts, masks, actions, deactivations, costs, incident_fracs = [], [], [], [], [], [], []

    while True:
        if policy_name == "rl":
            assert q_net is not None
            action = select_action(q_net, state, env.max_budget, epsilon=0.0, device=state.x.device).tolist()
        elif policy_name == "random":
            action = random_policy(state, env.max_budget)
        elif policy_name == "greedy":
            action = greedy_coverage_policy(env, state, env.max_budget)
        elif policy_name == "lookahead":
            action = lookahead_greedy_policy(
                env,
                state,
                env.max_budget,
                lookahead_h=lookahead_h,
                gamma=lookahead_gamma,
            )
        elif policy_name == "degree":
            action = static_degree_policy(env, state, env.max_budget)
        else:
            raise ValueError(policy_name)

        next_state, out = env.step(action)

        rewards.append(out.reward)
        counts.append(out.active_count)
        masks.append(env.active_mask.detach().cpu().numpy().copy())
        actions.append(list(action))
        deactivations.append(out.deactivation)
        costs.append(out.cost)
        incident_fracs.append(out.incident_fraction)

        state = next_state
        if out.done:
            break

    return summarize_episode(rewards, counts, masks, actions, deactivations, costs, incident_fracs)


def print_summary(title: str, policy_names: list[str], results: dict[str, list]) -> None:
    print(f"\n=== {title} ===")
    for name in policy_names:
        items = results[name]
        print(
            f"{name:>7s} | J={mean(m.cumulative_reward for m in items):7.3f} "
            f"| J_int={mean(m.j_int for m in items):7.3f} "
            f"| MTTD={mean(m.mean_time_to_deactivation for m in items):6.3f} "
            f"| S={mean(m.stability_score for m in items):6.3f} "
            f"| phi={mean(m.flicker_rate for m in items):6.3f} "
            f"| D={mean(m.avg_deactivation_count for m in items):6.3f} "
            f"| B={mean(m.avg_budget_used for m in items):6.3f}"
        )


def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)

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

    sample_state = env.reset(starts[0])
    q_net = GNNQNetwork(
        node_in_dim=sample_state.x.shape[1],
        edge_in_dim=sample_state.edge_attr.shape[1],
        hidden_dim=cfg.hidden_dim,
        gnn_layers=cfg.gnn_layers,
    ).to(device)

    best_model_path = cfg.workdir_path / "best_q_net.pt"
    model_path = best_model_path if best_model_path.exists() else (cfg.workdir_path / "q_net.pt")
    if model_path.exists():
        try:
            q_net.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        except RuntimeError as exc:
            print(
                "Warning: trained model is incompatible with the current "
                f"architecture ({exc}). RL evaluation will use random weights."
            )
    else:
        print("Warning: trained model not found. RL evaluation will use random weights.")

    policies = ["random", "greedy", "lookahead", "degree", "rl", "storm"]
    shared_schedule_policies = ["random", "greedy", "lookahead", "degree", "rl"]
    stress_test_policies = ["storm"]
    results = {name: [] for name in policies}

    real_incident_mode = (
        dataset.timestamps is not None
        and cfg.incident_events_xlsx is not None
        and cfg.incident_metadata_csv is not None
        and Path(cfg.incident_events_xlsx).exists()
        and Path(cfg.incident_metadata_csv).exists()
    )
    mapped_incidents = None
    if real_incident_mode:
        incident_map = map_incidents_to_segments(
            incidents_xlsx=cfg.incident_events_xlsx,
            metadata_csv=cfg.incident_metadata_csv,
            both_factor=cfg.incident_both_directions_factor,
            single_factor=cfg.incident_single_direction_factor,
        )
        mapped_incidents = incident_map.mapped
        report_paths = write_incident_mapping_reports(
            mapping=incident_map,
            metadata_csv=cfg.incident_metadata_csv,
            output_dir=cfg.workdir_path / "incident_reports_eval",
        )
        print(
            "Loaded incident events: "
            f"mapped_rows={incident_map.mapped_rows} unmapped_rows={incident_map.unmapped_rows}"
        )
        print(f"Incident mapping reports saved to {report_paths['summary_json']}")
    else:
        print("Incident file/metadata not available or timestamps missing; using sampled schedules.")

    for start_t in starts[: cfg.eval_episodes]:
        if mapped_incidents is not None:
            shared_schedule = build_incident_schedule_for_episode(
                start_t=start_t,
                horizon=cfg.horizon,
                num_edges=dataset.num_edges,
                timestamps=dataset.timestamps,
                mapped_incidents=mapped_incidents,
            )
        else:
            shared_schedule = env.sample_incident_schedule()
        for name in policies:
            policy_schedule = None if name == "storm" else shared_schedule
            metrics = run_episode(
                env,
                name,
                q_net,
                start_t,
                incident_schedule=policy_schedule,
                lookahead_h=cfg.greedy_lookahead_h,
                lookahead_gamma=cfg.greedy_lookahead_gamma,
            )
            results[name].append(metrics)

    print_summary("Shared-Schedule Policy Comparison", shared_schedule_policies, results)
    print_summary("Storm Stress Test", stress_test_policies, results)


if __name__ == "__main__":
    main()
