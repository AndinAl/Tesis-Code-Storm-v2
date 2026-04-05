from __future__ import annotations

import random
from typing import List

import torch

from .environment import CapacityConstrainedEnv, GraphState


def _normalize_score(score: torch.Tensor) -> torch.Tensor:
    if score.numel() == 0:
        return score
    score = score.float()
    spread = score.max() - score.min()
    if float(spread.item()) < 1e-6:
        return torch.zeros_like(score)
    return (score - score.min()) / spread


def _dynamic_prefix_from_score(score: torch.Tensor, max_budget: int, cost_weight: float) -> List[int]:
    limit = min(max_budget, score.numel())
    if limit <= 0:
        return []

    score = _normalize_score(score)
    sorted_score, sorted_idx = torch.sort(score, descending=True)
    prefix_gain = torch.cumsum(sorted_score[:limit], dim=0)
    cost = cost_weight * torch.arange(1, limit + 1, dtype=score.dtype)
    candidate_values = torch.cat([torch.zeros(1, dtype=score.dtype), prefix_gain - cost], dim=0)
    best_budget = int(torch.argmax(candidate_values).item())
    return sorted_idx[:best_budget].tolist()


def random_policy(state: GraphState, max_budget: int) -> List[int]:
    n = state.x.shape[0]
    budget = random.randint(0, min(max_budget, n))
    return random.sample(range(n), k=budget)


def greedy_coverage_policy(env: CapacityConstrainedEnv, state: GraphState, max_budget: int) -> List[int]:
    """Coverage-focused one-step heuristic.

    Score each candidate source node by how much one-hop effective outbound
    influence it can push under the current flow and dynamic capacity.
    """
    src, dst = env.dataset.edge_index
    flow_t = env.dataset.flows[env.current_t]
    dyn_cap = env._current_dynamic_capacity()
    eff = torch.minimum(flow_t, dyn_cap)

    score = torch.zeros(env.dataset.num_nodes, dtype=torch.float32)
    immediate_gain = eff / torch.clamp(env.static.thresholds[dst], min=1e-6)
    score.index_add_(0, src, immediate_gain)

    return _dynamic_prefix_from_score(score, max_budget, env.reward_zeta)


def static_degree_policy(env: CapacityConstrainedEnv, state: GraphState, max_budget: int) -> List[int]:
    score = env.dataset.outdegree
    return _dynamic_prefix_from_score(score, max_budget, env.reward_zeta)


def _simulate_discounted_return(
    env: CapacityConstrainedEnv,
    first_action: List[int],
    lookahead_h: int,
    gamma: float,
) -> float:
    """Approximate short-horizon return for one candidate intervention.

    We apply the candidate action at the current step and then roll forward
    with no new interventions for the remaining look-ahead steps.
    """
    src, dst = env.dataset.edge_index
    has_inbound = (env.static.inbound_capacity > 0).float()

    sim_active_mask = env.active_mask.clone()
    sim_visited_mask = env.visited_mask.clone()
    sim_prev_inbound_ratio = env.prev_inbound_ratio.clone()
    sim_current_t = int(env.current_t)
    sim_step_idx = int(env.step_idx)

    total = 0.0
    discount = 1.0

    horizon = max(1, int(lookahead_h))
    for h in range(horizon):
        if sim_step_idx >= env.horizon or sim_current_t >= env.dataset.num_steps:
            break

        action_idx = first_action if h == 0 else []
        if len(action_idx) > env.max_budget:
            action_idx = action_idx[: env.max_budget]

        flow_t = env.dataset.flows[sim_current_t]
        cap_scale = env.episode_capacity_scale[min(sim_step_idx, env.horizon - 1)]
        dyn_cap = env.static.capacity * cap_scale

        seed_mask = torch.zeros(env.dataset.num_nodes, dtype=torch.float32)
        if action_idx:
            seed_mask[torch.tensor(action_idx, dtype=torch.long)] = 1.0

        source_active = torch.maximum(sim_active_mask, seed_mask)
        active_edge_flow = torch.minimum(flow_t, dyn_cap) * source_active[src]
        inbound = torch.zeros(env.dataset.num_nodes, dtype=torch.float32)
        inbound.index_add_(0, dst, active_edge_flow)

        next_active = ((inbound >= env.static.thresholds) * has_inbound).float()
        next_active = torch.maximum(next_active, seed_mask)

        prev_active_mask = sim_active_mask
        coverage = float(((next_active > 0) & (sim_visited_mask == 0)).sum().item())
        persistence = float(((prev_active_mask > 0) & (next_active > 0)).sum().item())
        deactivation = float(((prev_active_mask > 0) & (next_active == 0)).sum().item())
        active_count = float(next_active.sum().item())
        cost = float(seed_mask.sum().item())

        diff = sim_prev_inbound_ratio - env.reward_saturation_threshold
        saturation_penalty = float(
            torch.where(
                diff > 0,
                (diff ** 2) * env.reward_saturation_weight * 10.0,
                torch.zeros_like(diff),
            )
            .mean()
            .item()
        )
        reward = env._compute_reward(
            coverage=coverage,
            persistence=persistence,
            deactivation=deactivation,
            saturation_penalty=saturation_penalty,
            active_count=active_count,
            cost=cost,
        )
        total += discount * float(reward)
        discount *= float(gamma)

        sim_visited_mask = torch.maximum(sim_visited_mask, next_active)
        sim_active_mask = next_active
        sim_prev_inbound_ratio = inbound / torch.clamp(env.static.inbound_capacity, min=1e-6)
        sim_current_t += 1
        sim_step_idx += 1

    return float(total)


def lookahead_greedy_policy(
    env: CapacityConstrainedEnv,
    state: GraphState,
    max_budget: int,
    lookahead_h: int = 6,
    gamma: float = 0.99,
) -> List[int]:
    """Look-ahead variant of greedy baseline for stricter comparison.

    Candidate sets are greedy prefixes of current one-hop influence ranking.
    We select the prefix that maximizes simulated discounted return over H.
    """
    del state
    src, dst = env.dataset.edge_index
    flow_t = env.dataset.flows[env.current_t]
    dyn_cap = env._current_dynamic_capacity()
    eff = torch.minimum(flow_t, dyn_cap)

    score = torch.zeros(env.dataset.num_nodes, dtype=torch.float32)
    immediate_gain = eff / torch.clamp(env.static.thresholds[dst], min=1e-6)
    score.index_add_(0, src, immediate_gain)
    score = _normalize_score(score)

    limit = min(max_budget, score.numel())
    if limit <= 0:
        return []

    _, sorted_idx = torch.sort(score, descending=True)
    candidates = [sorted_idx[:b].tolist() for b in range(0, limit + 1)]

    best_action: List[int] = []
    best_value = -float("inf")
    for action in candidates:
        value = _simulate_discounted_return(env, action, lookahead_h=lookahead_h, gamma=gamma)
        if value > best_value:
            best_value = value
            best_action = action
    return best_action
