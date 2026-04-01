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
