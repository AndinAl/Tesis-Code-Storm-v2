from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from .baselines import greedy_coverage_policy, random_policy, static_degree_policy
from .config import Config
from .data_loader import STIMDataset, build_dataset
from .environment import CapacityConstrainedEnv
from .evaluate import make_storm_schedule
from .incidents import build_incident_schedule_for_episode, map_incidents_to_segments
from .metrics import EpisodeMetrics, summarize_episode
from .model import GNNQNetwork, select_action
from .preprocess import StaticParameters, build_static_parameters


MM_PER_INCH = 25.4
HALF_WIDTH_IN = 85.0 / MM_PER_INCH
FULL_WIDTH_IN = 170.0 / MM_PER_INCH
MAX_HEIGHT_IN = 225.0 / MM_PER_INCH


@dataclass
class TraceStep:
    t: int
    reward: float
    cumulative_reward: float
    coverage: float
    persistence: float
    deactivation: float
    deactivation_penalty: float
    saturation_penalty: float
    cost: float
    active_count: float
    incident_fraction: float
    action: List[int]
    active_mask: np.ndarray
    utilization: np.ndarray
    utilization_c: np.ndarray
    utilization_d: np.ndarray
    edge_utilization: np.ndarray
    incident_edge_mask: np.ndarray
    saturation_overage: np.ndarray


@dataclass
class EpisodeTrace:
    policy_name: str
    start_t: int
    steps: List[TraceStep]
    metrics: EpisodeMetrics
    capacity_scale: np.ndarray


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "normal",
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_pdf(fig: plt.Figure, out_path: Path) -> None:
    fig.savefig(out_path, format="pdf", bbox_inches="tight")


def fresh_output_path(out_path: Path) -> Path:
    candidate = out_path
    version = 2
    while candidate.exists():
        candidate = out_path.with_name(f"{out_path.stem}_v{version}{out_path.suffix}")
        version += 1
    return candidate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clone_dataset_with_flow_scale(dataset: STIMDataset, flow_scale: float) -> STIMDataset:
    scaled_flows = dataset.flows.clone() * float(flow_scale)
    return STIMDataset(
        node_names=list(dataset.node_names),
        edge_names=list(dataset.edge_names),
        node_to_idx=dict(dataset.node_to_idx),
        edge_index=dataset.edge_index.clone(),
        flows=scaled_flows,
        distances=dataset.distances.clone(),
        coords=dataset.coords.clone(),
        community_ids=dataset.community_ids.clone(),
        gateway_scores=dataset.gateway_scores.clone(),
        indegree=dataset.indegree.clone(),
        outdegree=dataset.outdegree.clone(),
        edge_directions=list(dataset.edge_directions) if dataset.edge_directions is not None else None,
        timestamps=dataset.timestamps.copy() if dataset.timestamps is not None else None,
    )


def build_env(cfg: Config, dataset: STIMDataset, static: StaticParameters, device: str) -> CapacityConstrainedEnv:
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


def load_model(cfg: Config, env: CapacityConstrainedEnv, device: str) -> GNNQNetwork:
    sample_state = env.reset(env.static.val_starts[0])
    q_net = GNNQNetwork(
        node_in_dim=sample_state.x.shape[1],
        edge_in_dim=sample_state.edge_attr.shape[1],
        hidden_dim=cfg.hidden_dim,
        gnn_layers=cfg.gnn_layers,
    ).to(device)
    model_path = cfg.workdir_path / "q_net.pt"
    if model_path.exists():
        try:
            q_net.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        except RuntimeError as exc:
            print(
                "Warning: trained model is incompatible with the current "
                f"architecture ({exc}). Report generation will use random weights."
            )
    else:
        print("Warning: trained model not found. Report generation will use random weights.")
    q_net.eval()
    return q_net


def choose_action(
    env: CapacityConstrainedEnv,
    state,
    policy_name: str,
    q_net: GNNQNetwork | None,
) -> List[int]:
    if policy_name == "rl":
        assert q_net is not None
        return select_action(q_net, state, env.max_budget, epsilon=0.0, device=state.x.device).tolist()
    if policy_name == "random":
        return random_policy(state, env.max_budget)
    if policy_name == "greedy":
        return greedy_coverage_policy(env, state, env.max_budget)
    if policy_name == "degree":
        return static_degree_policy(env, state, env.max_budget)
    raise ValueError(policy_name)


def _direction_masks(dataset: STIMDataset) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    num_edges = dataset.num_edges
    dirs = dataset.edge_directions
    if dirs is not None and len(dirs) == num_edges:
        c_mask = torch.tensor([str(d).upper().startswith("C") for d in dirs], dtype=torch.bool)
        d_mask = torch.tensor([str(d).upper().startswith("D") for d in dirs], dtype=torch.bool)
        if c_mask.any() or d_mask.any():
            return c_mask, d_mask

    # Fallback for legacy datasets without explicit direction labels.
    lookup = {edge: i for i, edge in enumerate(dataset.edge_names)}
    c_mask = torch.zeros(num_edges, dtype=torch.bool)
    d_mask = torch.zeros(num_edges, dtype=torch.bool)
    for idx, (u, v) in enumerate(dataset.edge_names):
        rev = lookup.get((v, u))
        if rev is not None:
            if idx <= rev:
                c_mask[idx] = True
                d_mask[rev] = True
        else:
            c_mask[idx] = True
    return c_mask, d_mask


def run_episode_trace(
    env: CapacityConstrainedEnv,
    policy_name: str,
    q_net: GNNQNetwork | None,
    start_t: int,
    incident_schedule: torch.FloatTensor | None = None,
    reward_norm_edges: int | None = None,
) -> EpisodeTrace:
    if policy_name == "storm":
        state = env.reset(start_t, incident_schedule=make_storm_schedule(env, factor=0.6))
        inner_policy = "rl"
    else:
        state = env.reset(start_t, incident_schedule=incident_schedule)
        inner_policy = policy_name

    src = env.dataset.edge_index[0]
    dst = env.dataset.edge_index[1]
    dir_c_mask, dir_d_mask = _direction_masks(env.dataset)
    dir_c_weight = dir_c_mask.float()
    dir_d_weight = dir_d_mask.float()

    rewards: List[float] = []
    counts: List[float] = []
    masks: List[np.ndarray] = []
    actions: List[List[int]] = []
    deactivations: List[float] = []
    costs: List[float] = []
    incident_fracs: List[float] = []
    steps: List[TraceStep] = []
    cumulative_reward = 0.0
    reward_scale = 1.0
    if reward_norm_edges is not None and int(reward_norm_edges) > 0:
        reward_scale = 1.0 / float(int(reward_norm_edges))

    while True:
        t_before = env.current_t
        step_before = env.step_idx
        prev_active_mask = env.active_mask.clone()
        action = choose_action(env, state, inner_policy, q_net)
        next_state, out = env.step(action)

        step_reward = out.reward * reward_scale
        cumulative_reward += step_reward
        rewards.append(step_reward)
        counts.append(out.active_count)
        masks.append(env.active_mask.detach().cpu().numpy().copy())
        actions.append(list(action))
        deactivations.append(out.deactivation)
        costs.append(out.cost)
        incident_fracs.append(out.incident_fraction)

        flow_t = env.dataset.flows[t_before]
        scale_t = env.episode_capacity_scale[min(step_before, env.horizon - 1)]
        dyn_cap = env.static.capacity * scale_t
        seed_mask = torch.zeros(env.dataset.num_nodes, dtype=torch.float32)
        if action:
            seed_mask[torch.tensor(action, dtype=torch.long)] = 1.0
        source_active = torch.maximum(prev_active_mask, seed_mask)
        active_edge_flow = torch.minimum(flow_t, dyn_cap) * source_active[src]
        edge_utilization = active_edge_flow / torch.clamp(dyn_cap, min=1e-6)

        inbound_c = torch.zeros(env.dataset.num_nodes, dtype=torch.float32)
        inbound_d = torch.zeros(env.dataset.num_nodes, dtype=torch.float32)
        inbound_cap_c = torch.zeros(env.dataset.num_nodes, dtype=torch.float32)
        inbound_cap_d = torch.zeros(env.dataset.num_nodes, dtype=torch.float32)
        inbound_c.index_add_(0, dst, active_edge_flow * dir_c_weight)
        inbound_d.index_add_(0, dst, active_edge_flow * dir_d_weight)
        inbound_cap_c.index_add_(0, dst, dyn_cap * dir_c_weight)
        inbound_cap_d.index_add_(0, dst, dyn_cap * dir_d_weight)
        utilization_c = inbound_c / torch.clamp(inbound_cap_c, min=1e-6)
        utilization_d = inbound_d / torch.clamp(inbound_cap_d, min=1e-6)

        utilization = env.prev_inbound_ratio.detach().cpu().numpy().copy()
        saturation_overage = np.maximum(0.0, utilization - env.reward_saturation_threshold)
        steps.append(
            TraceStep(
                t=env.step_idx - 1,
                reward=step_reward,
                cumulative_reward=cumulative_reward,
                coverage=out.coverage,
                persistence=out.persistence,
                deactivation=out.deactivation,
                deactivation_penalty=out.deactivation_penalty,
                saturation_penalty=out.saturation_penalty,
                cost=out.cost,
                active_count=out.active_count,
                incident_fraction=out.incident_fraction,
                action=list(action),
                active_mask=env.active_mask.detach().cpu().numpy().copy(),
                utilization=utilization,
                utilization_c=utilization_c.detach().cpu().numpy().copy(),
                utilization_d=utilization_d.detach().cpu().numpy().copy(),
                edge_utilization=edge_utilization.detach().cpu().numpy().copy(),
                incident_edge_mask=(scale_t < 0.999).detach().cpu().numpy().astype(bool),
                saturation_overage=saturation_overage,
            )
        )

        state = next_state
        if out.done:
            break

    metrics = summarize_episode(rewards, counts, masks, actions, deactivations, costs, incident_fracs)
    return EpisodeTrace(
        policy_name=policy_name,
        start_t=start_t,
        steps=steps,
        metrics=metrics,
        capacity_scale=env.episode_capacity_scale.detach().cpu().numpy().copy(),
    )


def collect_policy_runs(
    cfg: Config,
    dataset: STIMDataset,
    static: StaticParameters,
    q_net: GNNQNetwork,
    device: str,
    flow_scale: float = 1.0,
    include_storm: bool = True,
    mapped_incidents=None,
    reward_norm_edges: int | None = None,
) -> Dict[str, List[EpisodeTrace]]:
    scaled_dataset = clone_dataset_with_flow_scale(dataset, flow_scale)
    env = build_env(cfg, scaled_dataset, static, device)
    starts = static.val_starts if cfg.eval_split == "val" else static.test_starts
    policies = ["random", "greedy", "degree", "rl"] + (["storm"] if include_storm else [])
    results = {name: [] for name in policies}

    for start_t in starts[: cfg.eval_episodes]:
        if mapped_incidents is not None and scaled_dataset.timestamps is not None:
            shared_schedule = build_incident_schedule_for_episode(
                start_t=start_t,
                horizon=cfg.horizon,
                num_edges=scaled_dataset.num_edges,
                timestamps=scaled_dataset.timestamps,
                mapped_incidents=mapped_incidents,
            )
        else:
            shared_schedule = env.sample_incident_schedule()
        for name in policies:
            schedule = None if name == "storm" else shared_schedule
            trace = run_episode_trace(
                env,
                name,
                q_net,
                start_t,
                incident_schedule=schedule,
                reward_norm_edges=reward_norm_edges,
            )
            results[name].append(trace)

    return results


def trace_saturation_overage(trace: EpisodeTrace) -> float:
    values = [float(step.saturation_overage.mean()) for step in trace.steps]
    return float(np.mean(values)) if values else 0.0


def trace_saturation_rate(trace: EpisodeTrace, threshold: float) -> float:
    flags = [(step.utilization > threshold).mean() for step in trace.steps]
    return float(np.mean(flags)) if flags else 0.0


def trace_directional_saturation_rate(trace: EpisodeTrace, threshold: float, direction: str) -> float:
    if direction.upper() == "C":
        flags = [(step.utilization_c > threshold).mean() for step in trace.steps]
    else:
        flags = [(step.utilization_d > threshold).mean() for step in trace.steps]
    return float(np.mean(flags)) if flags else 0.0


def trace_directional_mean_utilization(trace: EpisodeTrace, direction: str) -> float:
    if direction.upper() == "C":
        vals = [float(np.mean(step.utilization_c)) for step in trace.steps]
    else:
        vals = [float(np.mean(step.utilization_d)) for step in trace.steps]
    return float(np.mean(vals)) if vals else 0.0


def _reverse_edge_pairs(dataset: STIMDataset) -> list[tuple[int, int]]:
    lookup = {edge: idx for idx, edge in enumerate(dataset.edge_names)}
    pairs: list[tuple[int, int]] = []
    for idx, (u, v) in enumerate(dataset.edge_names):
        rev = lookup.get((v, u))
        if rev is not None and idx < rev:
            pairs.append((idx, rev))
    return pairs


def trace_directional_saturation_skew(trace: EpisodeTrace, reverse_pairs: list[tuple[int, int]]) -> float:
    if not reverse_pairs or not trace.steps:
        return 0.0
    skew_values = []
    for step in trace.steps:
        edge_util = step.edge_utilization
        pair_diffs = [abs(float(edge_util[i]) - float(edge_util[j])) for i, j in reverse_pairs]
        if pair_diffs:
            skew_values.append(float(np.mean(pair_diffs)))
    return float(np.mean(skew_values)) if skew_values else 0.0


def aggregate_policy_metrics(
    traces: Dict[str, List[EpisodeTrace]],
    saturation_threshold: float,
    dataset: STIMDataset,
) -> Dict[str, Dict[str, float]]:
    reverse_pairs = _reverse_edge_pairs(dataset)
    out: Dict[str, Dict[str, float]] = {}
    for policy_name, items in traces.items():
        out[policy_name] = {
            "J": mean(t.metrics.cumulative_reward for t in items),
            "J_int": mean(t.metrics.j_int for t in items),
            "MTTD": mean(t.metrics.mean_time_to_deactivation for t in items),
            "phi": mean(t.metrics.flicker_rate for t in items),
            "D": mean(t.metrics.avg_deactivation_count for t in items),
            "B": mean(t.metrics.avg_budget_used for t in items),
            "stability_score": mean(t.metrics.stability_score for t in items),
            "saturation_overage": mean(trace_saturation_overage(t) for t in items),
            "saturation_rate": mean(trace_saturation_rate(t, saturation_threshold) for t in items),
            "saturation_rate_c": mean(trace_directional_saturation_rate(t, saturation_threshold, "C") for t in items),
            "saturation_rate_d": mean(trace_directional_saturation_rate(t, saturation_threshold, "D") for t in items),
            "util_mean_c": mean(trace_directional_mean_utilization(t, "C") for t in items),
            "util_mean_d": mean(trace_directional_mean_utilization(t, "D") for t in items),
            "saturation_skew": mean(trace_directional_saturation_skew(t, reverse_pairs) for t in items),
        }
    return out


def find_backbone_advantage(
    rl_trace: EpisodeTrace,
    greedy_trace: EpisodeTrace,
    static: StaticParameters,
    threshold: float,
) -> tuple[int, int]:
    rl_util = np.stack([step.utilization for step in rl_trace.steps], axis=0)
    gr_util = np.stack([step.utilization for step in greedy_trace.steps], axis=0)
    rl_active = np.stack([step.active_mask for step in rl_trace.steps], axis=0)
    gr_active = np.stack([step.active_mask for step in greedy_trace.steps], axis=0)
    bet = static.betweenness_norm.detach().cpu().numpy()
    candidate_nodes = np.where(bet >= np.quantile(bet, 0.75))[0]
    if candidate_nodes.size == 0:
        candidate_nodes = np.arange(len(bet))

    best_score = -1e9
    best_pair = (int(np.argmax(bet)), 0)
    for node_idx in candidate_nodes:
        for t in range(len(rl_trace.steps)):
            gr_over = max(0.0, float(gr_util[t, node_idx] - threshold))
            rl_over = max(0.0, float(rl_util[t, node_idx] - threshold))
            score = (gr_over - rl_over) + 0.35 * bet[node_idx]
            if gr_active[t, node_idx] > 0.5 and rl_active[t, node_idx] < 0.5:
                score += 0.5
            if score > best_score:
                best_score = score
                best_pair = (int(node_idx), int(t))
    return best_pair


def find_peak_incident_step(rl_trace: EpisodeTrace, greedy_trace: EpisodeTrace) -> int:
    rl_load = np.array(
        [step.saturation_penalty + float(np.mean(step.incident_edge_mask)) for step in rl_trace.steps]
    )
    gr_load = np.array(
        [step.saturation_penalty + float(np.mean(step.incident_edge_mask)) for step in greedy_trace.steps]
    )
    peak = int(np.argmax(gr_load + rl_load))
    return peak


def plot_pareto_frontier(
    metrics: Dict[str, Dict[str, float]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 3.9), constrained_layout=True)
    marker_map = {
        "random": "o",
        "degree": "s",
        "greedy": "D",
        "rl": "P",
        "storm": "^",
    }
    skew_vals = [float(v.get("saturation_skew", 0.0)) for v in metrics.values()]
    skew_min = min(skew_vals) if skew_vals else 0.0
    skew_max = max(skew_vals) if skew_vals else 1.0
    if abs(skew_max - skew_min) < 1e-9:
        skew_max = skew_min + 1e-6
    norm = Normalize(vmin=skew_min, vmax=skew_max)
    cmap = plt.cm.magma

    points = []
    for name, vals in metrics.items():
        skew = float(vals.get("saturation_skew", 0.0))
        skew_n = float(norm(skew))
        points.append((name, vals["J_int"], vals["MTTD"]))
        ax.scatter(
            vals["J_int"],
            vals["MTTD"],
            s=120.0 + 170.0 * skew_n,
            marker=marker_map.get(name, "o"),
            color=cmap(skew_n),
            edgecolors="white",
            linewidths=0.8,
            label=name.upper(),
            zorder=3,
        )
        ax.annotate(
            name.upper(),
            xy=(vals["J_int"], vals["MTTD"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#cfcfcf", "alpha": 0.95},
        )

    frontier = []
    best_y = -1e9
    for item in sorted(points, key=lambda x: x[1]):
        if item[2] > best_y:
            frontier.append(item)
            best_y = item[2]
    if len(frontier) >= 2:
        ax.plot(
            [item[1] for item in frontier],
            [item[2] for item in frontier],
            color="#4d4d4d",
            linestyle="--",
            linewidth=1.6,
            alpha=0.8,
            label="Pareto-leading envelope",
            zorder=2,
        )

    ax.set_xlabel(r"Influence Volume $J_{int}$")
    ax.set_ylabel(r"Mean Time to Deactivation (MTTD)")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, loc="lower right")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label("Directional Saturation Skew")
    save_pdf(fig, out_path)
    plt.close(fig)


def plot_cumulative_reward(
    episode_runs: Dict[str, EpisodeTrace],
    out_path: Path,
    reward_alpha: float,
    reward_beta: float,
    reward_norm_edges: int | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 3.8), constrained_layout=True)
    palette = {
        "random": "#8c8c8c",
        "degree": "#2c7fb8",
        "greedy": "#d95f02",
        "rl": "#1b9e77",
        "storm": "#7570b3",
    }
    for name, trace in episode_runs.items():
        y = [step.cumulative_reward for step in trace.steps]
        x = np.arange(1, len(y) + 1)
        color = palette.get(name, "#333333")
        ax.plot(x, y, linewidth=2.4, color=color, label=name.upper())
        ax.scatter([x[-1]], [y[-1]], s=28, color=color, zorder=3)
        ax.annotate(
            name.upper(),
            xy=(x[-1], y[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=color,
        )

        if name == "rl":
            norm = float(int(reward_norm_edges)) if reward_norm_edges is not None and int(reward_norm_edges) > 0 else 1.0
            influence_cum = np.cumsum([(reward_alpha * step.coverage) / norm for step in trace.steps])
            persistence_cum = np.cumsum([(reward_beta * step.persistence) / norm for step in trace.steps])
            ax.fill_between(
                x,
                0.0,
                influence_cum,
                color="#9ecae1",
                alpha=0.20,
                label=r"RL $\alpha J_{int}$ component",
                zorder=1,
            )
            ax.fill_between(
                x,
                influence_cum,
                influence_cum + persistence_cum,
                color="#74c476",
                alpha=0.18,
                label=r"RL $\beta P$ component",
                zorder=1,
            )

    ax.set_xlabel("Snapshot within Episode")
    ax.set_ylabel(r"Cumulative Reward $J$")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    save_pdf(fig, out_path)
    plt.close(fig)


def plot_backbone_timeseries(
    rl_trace: EpisodeTrace,
    greedy_trace: EpisodeTrace,
    node_idx: int,
    moment_t: int,
    node_name: str,
    threshold: float,
    out_path: Path,
) -> None:
    rl_util_c = np.array([step.utilization_c[node_idx] for step in rl_trace.steps])
    gr_util_c = np.array([step.utilization_c[node_idx] for step in greedy_trace.steps])
    rl_util_d = np.array([step.utilization_d[node_idx] for step in rl_trace.steps])
    gr_util_d = np.array([step.utilization_d[node_idx] for step in greedy_trace.steps])
    rl_active = np.array([step.active_mask[node_idx] for step in rl_trace.steps])
    gr_active = np.array([step.active_mask[node_idx] for step in greedy_trace.steps])
    x = np.arange(len(rl_trace.steps), dtype=np.float32)

    fig, (ax_c, ax_d, ax_bottom) = plt.subplots(
        3,
        1,
        figsize=(FULL_WIDTH_IN, 5.6),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.0, 2.0, 1.2]},
    )

    ax_c.plot(x, gr_util_c, color="#d95f02", linewidth=2.1, label="Greedy (Dir C)")
    ax_c.plot(x, rl_util_c, color="#1b9e77", linewidth=2.1, label="RL (Dir C)")
    ax_c.axhline(threshold, color="#a50f15", linestyle="--", linewidth=1.3, label="Saturation Threshold")
    ax_c.axvline(moment_t, color="#444444", linestyle=":", linewidth=1.3)
    ax_c.fill_between(x, threshold, np.maximum(gr_util_c, threshold), where=gr_util_c > threshold, color="#d95f02", alpha=0.10)
    ax_c.set_ylabel("Utilization C")
    ax_c.text(
        0.01,
        0.97,
        f"Backbone node: {node_name}",
        transform=ax_c.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#d0d0d0", "alpha": 0.95},
    )
    ax_c.grid(alpha=0.25, linestyle="--")
    ax_c.legend(frameon=False, ncol=3, loc="upper left")

    ax_d.plot(x, gr_util_d, color="#e6550d", linewidth=2.1, linestyle="--", label="Greedy (Dir D)")
    ax_d.plot(x, rl_util_d, color="#31a354", linewidth=2.1, linestyle="--", label="RL (Dir D)")
    ax_d.axhline(threshold, color="#a50f15", linestyle="--", linewidth=1.3)
    ax_d.axvline(moment_t, color="#444444", linestyle=":", linewidth=1.3)
    ax_d.fill_between(x, threshold, np.maximum(gr_util_d, threshold), where=gr_util_d > threshold, color="#e6550d", alpha=0.08)
    ax_d.annotate(
        "Asymmetry handled by RL:\nDir C and Dir D can diverge",
        xy=(moment_t, rl_util_d[moment_t]),
        xytext=(max(0, moment_t - 5), max(threshold + 0.03, float(rl_util_d.max()) + 0.05)),
        arrowprops={"arrowstyle": "->", "color": "#1b9e77"},
        fontsize=8,
        color="#1b9e77",
    )
    ax_d.set_ylabel("Utilization D")
    ax_d.grid(alpha=0.25, linestyle="--")
    ax_d.legend(frameon=False, ncol=2, loc="upper left")

    ax_bottom.step(x, gr_active, where="post", color="#d95f02", linewidth=2.0, label="Greedy Active")
    ax_bottom.step(x, rl_active, where="post", color="#1b9e77", linewidth=2.0, label="RL Active")
    ax_bottom.fill_between(x, 0, rl_active, step="post", alpha=0.18, color="#1b9e77")
    ax_bottom.fill_between(x, 0, gr_active, step="post", alpha=0.12, color="#d95f02")
    ax_bottom.axvline(moment_t, color="#444444", linestyle=":", linewidth=1.5)
    ax_bottom.set_ylim(-0.05, 1.15)
    ax_bottom.set_yticks([0, 1])
    ax_bottom.set_yticklabels(["Off", "On"])
    ax_bottom.set_xlabel("Snapshot within Episode")
    ax_bottom.set_ylabel("Activation")
    ax_bottom.grid(alpha=0.25, linestyle="--")
    ax_bottom.legend(frameon=False, ncol=2, loc="upper left")
    save_pdf(fig, out_path)
    plt.close(fig)


def top_betweenness_indices(static: StaticParameters, top_k: int = 10) -> np.ndarray:
    bet = static.betweenness_norm.detach().cpu().numpy()
    order = np.argsort(-bet)
    return order[: min(top_k, len(order))]


def _directed_edge_segments(dataset: STIMDataset, offset_ratio: float = 0.08) -> np.ndarray:
    coords = dataset.coords.detach().cpu().numpy()
    src = dataset.edge_index[0].detach().cpu().numpy()
    dst = dataset.edge_index[1].detach().cpu().numpy()
    segments = np.zeros((dataset.num_edges, 2, 2), dtype=np.float32)
    for e_idx, (u, v) in enumerate(zip(src, dst)):
        x0 = float(coords[u, 1])
        y0 = float(coords[u, 0])
        x1 = float(coords[v, 1])
        y1 = float(coords[v, 0])
        dx = x1 - x0
        dy = y1 - y0
        norm = float(np.hypot(dx, dy))
        if norm > 1e-9:
            sign = 1.0 if str(dataset.node_names[u]) <= str(dataset.node_names[v]) else -1.0
            shift = offset_ratio * norm
            ox = -dy / norm * shift * sign
            oy = dx / norm * shift * sign
        else:
            ox = 0.0
            oy = 0.0
        segments[e_idx, 0, :] = [x0 + ox, y0 + oy]
        segments[e_idx, 1, :] = [x1 + ox, y1 + oy]
    return segments


def _add_segment_arrow(ax, segment: np.ndarray, color: str, alpha: float = 0.6, zorder: int = 3) -> None:
    x0, y0 = float(segment[0, 0]), float(segment[0, 1])
    x1, y1 = float(segment[1, 0]), float(segment[1, 1])
    xm0 = x0 + 0.45 * (x1 - x0)
    ym0 = y0 + 0.45 * (y1 - y0)
    xm1 = x0 + 0.62 * (x1 - x0)
    ym1 = y0 + 0.62 * (y1 - y0)
    ax.annotate(
        "",
        xy=(xm1, ym1),
        xytext=(xm0, ym0),
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": color, "alpha": alpha},
        zorder=zorder,
    )


def plot_base_map(
    ax,
    dataset: STIMDataset,
    static: StaticParameters,
    highlight_top10: bool = False,
    backbone_idx: int | None = None,
) -> None:
    coords = dataset.coords.detach().cpu().numpy()
    lat = coords[:, 0]
    lon = coords[:, 1]
    src = dataset.edge_index[0].detach().cpu().numpy()
    dst = dataset.edge_index[1].detach().cpu().numpy()
    bet = static.betweenness_norm.detach().cpu().numpy()
    edge_importance = 0.5 * (bet[src] + bet[dst])
    segments = _directed_edge_segments(dataset)
    edge_lines = LineCollection(
        segments,
        colors="#9e9e9e",
        linewidths=0.25 + 2.4 * edge_importance,
        alpha=0.35,
        zorder=1,
    )
    ax.add_collection(edge_lines)
    arrow_cut = float(np.quantile(edge_importance, 0.88)) if edge_importance.size > 0 else 1.0
    for e_idx in np.where(edge_importance >= arrow_cut)[0]:
        _add_segment_arrow(ax, segments[e_idx], color="#7a7a7a", alpha=0.55, zorder=2)
    ax.scatter(lon, lat, s=12, color="#6f6f6f", alpha=0.85, zorder=2)

    if highlight_top10:
        top_idx = top_betweenness_indices(static, top_k=10)
        ax.scatter(
            lon[top_idx],
            lat[top_idx],
            s=84,
            color="#fdb863",
            edgecolors="#4d4d4d",
            linewidths=0.8,
            zorder=4,
        )
        for rank, node_idx in enumerate(top_idx, start=1):
            ax.text(
                lon[node_idx],
                lat[node_idx],
                f"{rank}",
                fontsize=8,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "#4d4d4d", "alpha": 0.9},
                zorder=5,
            )

    if backbone_idx is not None:
        ax.scatter(
            lon[backbone_idx],
            lat[backbone_idx],
            s=170,
            marker="*",
            color="#111111",
            edgecolors="white",
            linewidths=0.8,
            zorder=6,
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.15, linestyle=":")


def overlay_policy_map(
    ax,
    dataset: STIMDataset,
    static: StaticParameters,
    trace: EpisodeTrace,
    t: int,
    backbone_idx: int,
    title: str,
    highlight_incidents: bool = False,
) -> LineCollection:
    plot_base_map(ax, dataset, static, highlight_top10=False, backbone_idx=backbone_idx)
    coords = dataset.coords.detach().cpu().numpy()
    lat = coords[:, 0]
    lon = coords[:, 1]
    step = trace.steps[t]
    prev_step = trace.steps[max(0, t - 1)]
    intensity = np.clip(step.edge_utilization, 0.0, 1.0)
    segments = _directed_edge_segments(dataset)

    color_edges = LineCollection(
        segments,
        array=intensity,
        cmap="YlOrRd",
        linewidths=0.5 + 2.6 * intensity,
        alpha=0.82,
        zorder=3,
    )
    ax.add_collection(color_edges)

    if highlight_incidents and bool(step.incident_edge_mask.any()):
        incident_segments = segments[step.incident_edge_mask]
        incident_lines = LineCollection(
            incident_segments,
            colors="#2b8cbe",
            linewidths=2.8,
            alpha=0.88,
            zorder=4,
        )
        ax.add_collection(incident_lines)
        for seg in incident_segments[:20]:
            _add_segment_arrow(ax, seg, color="#2b8cbe", alpha=0.9, zorder=5)
        ax.text(
            0.02,
            0.08,
            f"Incident-reduced directed segments: {int(step.incident_edge_mask.sum())}",
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#b5d4e9", "alpha": 0.96},
        )

    prev_active = prev_step.active_mask > 0.5
    curr_active = step.active_mask > 0.5

    if prev_active.any():
        ax.scatter(
            lon[prev_active],
            lat[prev_active],
            s=180,
            facecolors="none",
            edgecolors="#66c2ff",
            alpha=0.28,
            linewidths=1.8,
            zorder=3,
        )
        prev_centroid = np.array([lon[prev_active].mean(), lat[prev_active].mean()])
        curr_centroid = np.array([lon[curr_active].mean(), lat[curr_active].mean()]) if curr_active.any() else prev_centroid
        ax.annotate(
            "",
            xy=(curr_centroid[0], curr_centroid[1]),
            xytext=(prev_centroid[0], prev_centroid[1]),
            arrowprops={"arrowstyle": "->", "color": "#4f4f4f", "alpha": 0.4, "lw": 1.4},
            zorder=4,
        )

    if curr_active.any():
        ax.scatter(
            lon[curr_active],
            lat[curr_active],
            s=140,
            color="#1976d2",
            edgecolors="white",
            linewidths=0.9,
            zorder=5,
        )

    ax.text(
        0.02,
        0.98,
        title,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#d0d0d0", "alpha": 0.95},
    )
    return color_edges


def create_betweenness_pdf(
    dataset: STIMDataset,
    static: StaticParameters,
    backbone_idx: int,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 4.6), constrained_layout=True)
    plot_base_map(ax, dataset, static, highlight_top10=True, backbone_idx=backbone_idx)
    handles = [
        Line2D([0], [0], color="#9e9e9e", linewidth=2.0, label="Directed edges weighted by betweenness proxy"),
        Line2D([0], [0], color="#7a7a7a", linewidth=1.6, label="Small arrows indicate flow direction"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#6f6f6f", markersize=7, label="All nodes"),
        Line2D([0], [0], marker="o", color="w", markeredgecolor="#4d4d4d", markerfacecolor="#fdb863", markersize=9, label="Top-10 node betweenness"),
        Line2D([0], [0], marker="*", color="w", markeredgecolor="white", markerfacecolor="#111111", markersize=12, label="Selected backbone node"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left")
    save_pdf(fig, out_path)
    plt.close(fig)


def _map_legend_handles() -> List[Line2D]:
    return [
        Line2D([0], [0], color="#9e9e9e", linewidth=2.0, label="Directed edges weighted by static betweenness"),
        Line2D([0], [0], color="#f16913", linewidth=2.4, label="Directional edge utilization"),
        Line2D([0], [0], color="#2b8cbe", linewidth=2.6, label="Incident-reduced directed edge"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#6f6f6f", markersize=7, label="Nodes"),
        Line2D([0], [0], marker="o", color="w", markeredgecolor="#66c2ff", markerfacecolor="none", markersize=10, label="Previous active set"),
        Line2D([0], [0], marker="o", color="w", markeredgecolor="white", markerfacecolor="#1976d2", markersize=10, label="Current active set"),
        Line2D([0], [0], marker="*", color="w", markeredgecolor="white", markerfacecolor="#111111", markersize=12, label="Selected backbone node"),
    ]


def create_incident_trace_pdf(
    dataset: STIMDataset,
    static: StaticParameters,
    rl_trace: EpisodeTrace,
    backbone_idx: int,
    peak_t: int,
    out_path: Path,
) -> None:
    map_legend = _map_legend_handles()
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH_IN, 2.9), constrained_layout=True)
    times = [max(0, peak_t - 1), peak_t, min(len(rl_trace.steps) - 1, peak_t + 1)]
    contour: LineCollection | None = None
    for ax, t in zip(axes, times):
        contour = overlay_policy_map(
            ax,
            dataset,
            static,
            rl_trace,
            t,
            backbone_idx,
            title=f"RL (SMA), t={t}",
            highlight_incidents=True,
        )
    if contour is not None:
        fig.colorbar(contour, ax=axes, fraction=0.03, pad=0.02, label="Directed Edge Utilization")
    fig.legend(handles=map_legend, frameon=False, ncol=3, loc="lower center")
    save_pdf(fig, out_path)
    plt.close(fig)


def create_advantage_pdf(
    dataset: STIMDataset,
    static: StaticParameters,
    rl_trace: EpisodeTrace,
    greedy_trace: EpisodeTrace,
    backbone_idx: int,
    peak_t: int,
    out_path: Path,
) -> None:
    map_legend = _map_legend_handles()
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 3.2), constrained_layout=True)
    contour_left = overlay_policy_map(
        axes[0],
        dataset,
        static,
        rl_trace,
        peak_t,
        backbone_idx,
        title=f"RL (SMA), t={peak_t}",
        highlight_incidents=True,
    )
    contour_right = overlay_policy_map(
        axes[1],
        dataset,
        static,
        greedy_trace,
        peak_t,
        backbone_idx,
        title=f"Greedy, t={peak_t}",
        highlight_incidents=True,
    )
    contour = contour_right if contour_right is not None else contour_left
    fig.colorbar(contour, ax=axes, fraction=0.03, pad=0.02, label="Directed Edge Utilization")
    fig.legend(handles=map_legend, frameon=False, ncol=3, loc="lower center")
    save_pdf(fig, out_path)
    plt.close(fig)


def format_float(value: float) -> str:
    return f"{value:.3f}"


def write_latex_tables(
    metrics: Dict[str, Dict[str, float]],
    stress_rows: List[dict[str, float]],
    out_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("% Auto-generated LaTeX tables saved as text for paper integration.")
    lines.append("")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Directional multi-objective performance summary on the shared-schedule validation split. $S$ is average saturation overage, $\\rho_C/\\rho_D$ are directional saturation rates, and $\\Delta_{dir}$ is directional saturation skew.}")
    lines.append("\\label{tab:multi_objective_summary}")
    lines.append("\\begin{tabular}{lccccccc}")
    lines.append("\\toprule")
    lines.append("Policy & $J$ & $J_{int}$ & MTTD & $\\phi$ & $S$ & $\\rho_C/\\rho_D$ & $\\Delta_{dir}$ \\\\")
    lines.append("\\midrule")
    for policy_name in ["random", "degree", "greedy", "rl"]:
        vals = metrics[policy_name]
        label = "RL (SMA)" if policy_name == "rl" else policy_name.capitalize()
        lines.append(
            f"{label} & {format_float(vals['J'])} & {format_float(vals['J_int'])} & "
            f"{format_float(vals['MTTD'])} & {format_float(vals['phi'])} & "
            f"{format_float(vals['saturation_overage'])} & "
            f"{format_float(vals['saturation_rate_c'])}/{format_float(vals['saturation_rate_d'])} & "
            f"{format_float(vals['saturation_skew'])} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Sensitivity to incident intensity under directional flow asymmetry. $\\Delta J$ is RL minus baseline retention; $\\Delta_{skew}$ is baseline minus RL directional skew.}")
    lines.append("\\label{tab:stress_test}")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append("Incident Scale & RL $J_{int}$ Retention & $\\Delta J$ & RL Skew & Baseline Skew & $\\Delta_{skew}$ \\\\")
    lines.append("\\midrule")
    for row in stress_rows:
        lines.append(
            f"{row['label']} & {row['rl_retention']:.1f}\\% & {row['delta']:+.1f}\\% & "
            f"{row['rl_skew']:.3f} & {row['baseline_skew']:.3f} & {row['skew_delta']:+.3f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_metrics_csv(metrics: Dict[str, Dict[str, float]], out_path: Path) -> None:
    header = (
        "policy,J,J_int,MTTD,phi,D,B,stability_score,"
        "saturation_overage,saturation_rate,saturation_rate_c,saturation_rate_d,"
        "util_mean_c,util_mean_d,saturation_skew"
    )
    rows = [header]
    for policy_name, vals in metrics.items():
        rows.append(
            ",".join(
                [
                    policy_name,
                    format_float(vals["J"]),
                    format_float(vals["J_int"]),
                    format_float(vals["MTTD"]),
                    format_float(vals["phi"]),
                    format_float(vals["D"]),
                    format_float(vals["B"]),
                    format_float(vals["stability_score"]),
                    format_float(vals["saturation_overage"]),
                    format_float(vals["saturation_rate"]),
                    format_float(vals["saturation_rate_c"]),
                    format_float(vals["saturation_rate_d"]),
                    format_float(vals["util_mean_c"]),
                    format_float(vals["util_mean_d"]),
                    format_float(vals["saturation_skew"]),
                ]
            )
        )
    out_path.write_text("\n".join(rows), encoding="utf-8")


def build_stress_test_rows(
    cfg: Config,
    dataset: STIMDataset,
    static: StaticParameters,
    q_net: GNNQNetwork,
    device: str,
    mapped_incidents=None,
) -> tuple[List[dict[str, float]], Dict[str, Dict[str, Dict[str, float]]]]:
    scales = [1.0, 2.0, 3.0]
    labels = {1.0: "Normal (1.0x)", 2.0: "High (2.0x)", 3.0: "Extreme (3.0x)"}
    per_scale_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for scale in scales:
        traces = collect_policy_runs(
            cfg,
            dataset,
            static,
            q_net,
            device,
            flow_scale=scale,
            include_storm=False,
            mapped_incidents=mapped_incidents,
            reward_norm_edges=cfg.report_reward_normalization_edges,
        )
        per_scale_metrics[str(scale)] = aggregate_policy_metrics(traces, cfg.reward_saturation_threshold, dataset)

    rl_base = per_scale_metrics["1.0"]["rl"]["J_int"]
    greedy_base = per_scale_metrics["1.0"]["greedy"]["J_int"]
    rows: List[dict[str, float]] = []
    for scale in scales:
        metrics = per_scale_metrics[str(scale)]
        rl_retention = 100.0 * metrics["rl"]["J_int"] / max(rl_base, 1e-6)
        baseline_retention = 100.0 * metrics["greedy"]["J_int"] / max(greedy_base, 1e-6)
        rl_skew = float(metrics["rl"]["saturation_skew"])
        baseline_skew = float(metrics["greedy"]["saturation_skew"])
        rows.append(
            {
                "label": labels[scale],
                "rl_retention": rl_retention,
                "baseline_retention": baseline_retention,
                "delta": rl_retention - baseline_retention,
                "rl_skew": rl_skew,
                "baseline_skew": baseline_skew,
                "skew_delta": baseline_skew - rl_skew,
            }
        )
    return rows, per_scale_metrics


def main() -> None:
    cfg = Config()
    set_publication_style()
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    outdir = cfg.workdir_path / "paper_assets"
    outdir.mkdir(parents=True, exist_ok=True)

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
    env = build_env(cfg, dataset, static, device)
    q_net = load_model(cfg, env, device)

    mapped_incidents = None
    if (
        dataset.timestamps is not None
        and cfg.incident_events_xlsx is not None
        and cfg.incident_metadata_csv is not None
        and Path(cfg.incident_events_xlsx).exists()
        and Path(cfg.incident_metadata_csv).exists()
    ):
        incident_map = map_incidents_to_segments(
            incidents_xlsx=cfg.incident_events_xlsx,
            metadata_csv=cfg.incident_metadata_csv,
            both_factor=cfg.incident_both_directions_factor,
            single_factor=cfg.incident_single_direction_factor,
        )
        mapped_incidents = incident_map.mapped

    traces = collect_policy_runs(
        cfg,
        dataset,
        static,
        q_net,
        device,
        flow_scale=1.0,
        include_storm=True,
        mapped_incidents=mapped_incidents,
        reward_norm_edges=cfg.report_reward_normalization_edges,
    )
    metrics = aggregate_policy_metrics(traces, cfg.reward_saturation_threshold, dataset)

    pareto_path = fresh_output_path(outdir / "pareto_frontier.pdf")
    plot_pareto_frontier(metrics, pareto_path)

    shared_episode_runs = {name: traces[name][0] for name in ["random", "degree", "greedy", "rl", "storm"]}
    cumulative_path = fresh_output_path(outdir / "cumulative_reward_comparison.pdf")
    plot_cumulative_reward(
        shared_episode_runs,
        cumulative_path,
        reward_alpha=cfg.reward_alpha,
        reward_beta=cfg.reward_beta,
        reward_norm_edges=cfg.report_reward_normalization_edges,
    )

    backbone_idx, moment_t = find_backbone_advantage(
        rl_trace=traces["rl"][0],
        greedy_trace=traces["greedy"][0],
        static=static,
        threshold=cfg.reward_saturation_threshold,
    )
    backbone_name = dataset.node_names[backbone_idx]
    timeseries_path = fresh_output_path(outdir / "backbone_utilization_vs_capacity.pdf")
    plot_backbone_timeseries(
        rl_trace=traces["rl"][0],
        greedy_trace=traces["greedy"][0],
        node_idx=backbone_idx,
        moment_t=moment_t,
        node_name=backbone_name,
        threshold=cfg.reward_saturation_threshold,
        out_path=timeseries_path,
    )
    betweenness_path = fresh_output_path(outdir / "node_betweenness_top10.pdf")
    create_betweenness_pdf(
        dataset=dataset,
        static=static,
        backbone_idx=backbone_idx,
        out_path=betweenness_path,
    )

    geodash_rl_path = fresh_output_path(outdir / "geodash_rl_incident_trace.pdf")
    geodash_adv_path = fresh_output_path(outdir / "geodash_rl_vs_greedy_peak.pdf")
    peak_t = find_peak_incident_step(traces["rl"][0], traces["greedy"][0])
    create_incident_trace_pdf(
        dataset=dataset,
        static=static,
        rl_trace=traces["rl"][0],
        backbone_idx=backbone_idx,
        peak_t=peak_t,
        out_path=geodash_rl_path,
    )
    create_advantage_pdf(
        dataset=dataset,
        static=static,
        rl_trace=traces["rl"][0],
        greedy_trace=traces["greedy"][0],
        backbone_idx=backbone_idx,
        peak_t=peak_t,
        out_path=geodash_adv_path,
    )

    stress_rows, per_scale_metrics = build_stress_test_rows(
        cfg,
        dataset,
        static,
        q_net,
        device,
        mapped_incidents=mapped_incidents,
    )
    latex_path = outdir / "analysis_tables.tex.txt"
    write_latex_tables(metrics, stress_rows, latex_path)

    write_metrics_csv(metrics, outdir / "latest_policy_metrics.csv")
    stress_lines = [
        "scale,policy,J_int,MTTD,saturation_rate,saturation_rate_c,saturation_rate_d,util_mean_c,util_mean_d,saturation_skew"
    ]
    for scale, scale_metrics in per_scale_metrics.items():
        for policy_name in ["greedy", "rl"]:
            vals = scale_metrics[policy_name]
            stress_lines.append(
                ",".join(
                    [
                        str(scale),
                        str(policy_name),
                        format_float(vals["J_int"]),
                        format_float(vals["MTTD"]),
                        format_float(100.0 * vals["saturation_rate"]),
                        format_float(100.0 * vals["saturation_rate_c"]),
                        format_float(100.0 * vals["saturation_rate_d"]),
                        format_float(vals["util_mean_c"]),
                        format_float(vals["util_mean_d"]),
                        format_float(vals["saturation_skew"]),
                    ]
                )
            )
    (outdir / "stress_test_metrics.csv").write_text("\n".join(stress_lines), encoding="utf-8")

    summary_lines = [
        f"Backbone node: {backbone_name} (index={backbone_idx})",
        f"Annotated advantage moment: t={moment_t}",
        f"Peak incident map step: t={peak_t}",
        f"Pareto plot: {pareto_path}",
        f"Cumulative reward plot: {cumulative_path}",
        f"Backbone time-series plot: {timeseries_path}",
        f"Node betweenness top-10 plot: {betweenness_path}",
        f"RL incident trace geodash: {geodash_rl_path}",
        f"RL vs Greedy peak geodash: {geodash_adv_path}",
        f"LaTeX tables: {latex_path}",
    ]
    (outdir / "report_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
