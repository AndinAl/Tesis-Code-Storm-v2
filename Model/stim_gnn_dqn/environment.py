from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import torch

from .data_loader import STIMDataset
from .preprocess import StaticParameters


def normalize_incident_amplitude(amplitude: object) -> str:
    return str(amplitude).strip().upper()


def incident_amplitude_to_directions(amplitude: object) -> set[str]:
    text = normalize_incident_amplitude(amplitude)
    if "AMBOS" in text:
        return {"C", "D"}
    if "DECRESCENTE" in text:
        return {"D"}
    if "CRESCENTE" in text:
        return {"C"}
    return {"C", "D"}


def incident_text_to_directions(text: object) -> set[str] | None:
    value = normalize_incident_amplitude(text)
    if not value or value in {"NAN", "NONE"}:
        return None
    if "AMBOS" in value:
        return {"C", "D"}
    if "DECRESCENTE" in value:
        return {"D"}
    if "CRESCENTE" in value:
        return {"C"}
    if "SENTIDO PORTO ALEGRE - CANOAS" in value:
        return {"C"}
    if "SENTIDO CANOAS - PORTO ALEGRE" in value:
        return {"D"}
    return None


@dataclass
class GraphState:
    x: torch.FloatTensor
    edge_index: torch.LongTensor
    edge_attr: torch.FloatTensor
    global_t: int

    def clone(self) -> "GraphState":
        return GraphState(
            x=self.x.clone().detach(),
            edge_index=self.edge_index.clone().detach(),
            edge_attr=self.edge_attr.clone().detach(),
            global_t=int(self.global_t),
        )


@dataclass
class StepOutput:
    reward: float
    coverage: float
    persistence: float
    deactivation: float
    deactivation_penalty: float
    saturation_penalty: float
    cost: float
    active_count: float
    done: bool
    incident_fraction: float


class CapacityConstrainedEnv:
    def __init__(
        self,
        dataset: STIMDataset,
        static: StaticParameters,
        horizon: int,
        max_budget: int,
        history_len: int,
        reward_alpha: float,
        reward_beta: float,
        reward_kappa: float,
        reward_eta: float,
        reward_deactivation_lambda: float,
        reward_saturation_threshold: float,
        reward_saturation_weight: float,
        reward_zeta: float,
        incident_prob: float,
        incident_factor_low: float,
        incident_factor_high: float,
        incident_duration_min: int,
        incident_duration_max: int,
        incident_edge_fraction: float,
        device: str = "cpu",
    ) -> None:
        self.dataset = dataset
        self.static = static
        self.horizon = horizon
        self.max_budget = max_budget
        self.history_len = history_len
        self.reward_alpha = reward_alpha
        self.reward_beta = reward_beta
        self.reward_kappa = reward_kappa
        self.reward_eta = reward_eta
        self.reward_deactivation_lambda = reward_deactivation_lambda
        self.reward_saturation_threshold = reward_saturation_threshold
        self.reward_saturation_weight = reward_saturation_weight
        self.reward_zeta = reward_zeta
        self.incident_prob = incident_prob
        self.incident_factor_low = incident_factor_low
        self.incident_factor_high = incident_factor_high
        self.incident_duration_min = incident_duration_min
        self.incident_duration_max = incident_duration_max
        self.incident_edge_fraction = incident_edge_fraction
        self.device = device

        self.current_t = 0
        self.step_idx = 0
        self.active_mask = torch.zeros(dataset.num_nodes, dtype=torch.float32)
        self.active_duration = torch.zeros(dataset.num_nodes, dtype=torch.float32)
        self.visited_mask = torch.zeros(dataset.num_nodes, dtype=torch.float32)
        self.prev_inbound_ratio = torch.zeros(dataset.num_nodes, dtype=torch.float32)
        self.utilization_history: Deque[torch.Tensor] = deque(
            [self.prev_inbound_ratio.clone()],
            maxlen=3,
        )
        self.episode_capacity_scale = torch.ones((horizon, dataset.num_edges), dtype=torch.float32)
        # Normalize coverage by graph scale.
        self.coverage_norm_factor = 1.0 / float(max(1, self.dataset.num_nodes))

    def _sample_incident_schedule(self) -> torch.FloatTensor:
        schedule = torch.ones((self.horizon, self.dataset.num_edges), dtype=torch.float32)

        if random.random() > self.incident_prob:
            return schedule

        duration = random.randint(self.incident_duration_min, self.incident_duration_max)
        start = random.randint(max(0, self.history_len), max(self.history_len, self.horizon - duration))
        edge_count = max(1, int(self.dataset.num_edges * self.incident_edge_fraction))

        cap_scores = self.static.capacity.clone()
        top_edges = torch.topk(cap_scores, k=min(edge_count, cap_scores.numel())).indices.tolist()
        affected = random.sample(top_edges, k=min(edge_count, len(top_edges)))
        factor = random.uniform(self.incident_factor_low, self.incident_factor_high)

        for t in range(start, min(self.horizon, start + duration)):
            schedule[t, affected] = factor
        return schedule

    def sample_incident_schedule(self) -> torch.FloatTensor:
        return self._sample_incident_schedule()

    def reset(self, start_t: int, incident_schedule: Optional[torch.FloatTensor] = None) -> GraphState:
        self.current_t = int(start_t)
        self.step_idx = 0
        self.active_mask = torch.zeros(self.dataset.num_nodes, dtype=torch.float32)
        self.active_duration = torch.zeros(self.dataset.num_nodes, dtype=torch.float32)
        self.visited_mask = torch.zeros(self.dataset.num_nodes, dtype=torch.float32)
        self.prev_inbound_ratio = torch.zeros(self.dataset.num_nodes, dtype=torch.float32)
        self.utilization_history = deque([self.prev_inbound_ratio.clone()], maxlen=3)
        self.episode_capacity_scale = (
            incident_schedule.clone().float()
            if incident_schedule is not None
            else self._sample_incident_schedule()
        )
        return self._build_state()

    def _history_tensor(self, flow_t: torch.Tensor) -> torch.Tensor:
        hist = []
        start = max(0, self.current_t - self.history_len + 1)
        for t in range(start, self.current_t + 1):
            hist.append(self.dataset.flows[t])
        while len(hist) < self.history_len:
            hist.insert(0, torch.zeros_like(flow_t))
        return torch.stack(hist, dim=0)

    def _seasonal_encoding(self) -> torch.Tensor:
        hour = self.current_t % self.horizon
        angle = 2.0 * torch.pi * torch.tensor(hour / self.horizon, dtype=torch.float32)
        return torch.tensor([torch.sin(angle), torch.cos(angle)], dtype=torch.float32)

    def _current_dynamic_capacity(self) -> torch.Tensor:
        scale = self.episode_capacity_scale[min(self.step_idx, self.horizon - 1)]
        return self.static.capacity * scale

    def _compute_saturation_penalty(self) -> float:
        diff = self.prev_inbound_ratio - self.reward_saturation_threshold
        stress_penalty = torch.where(
            diff > 0,
            (diff ** 2) * self.reward_saturation_weight * 10.0,
            torch.zeros_like(diff),
        ).mean()
        return float(stress_penalty.item())

    def _compute_reward(
        self,
        coverage: float,
        persistence: float,
        deactivation: float,
        saturation_penalty: float,
        active_count: float,
        cost: float,
    ) -> float:
        # Retained for interface compatibility with step() bookkeeping.
        del cost
        norm_coverage = coverage * self.coverage_norm_factor
        persistence_norm = persistence / max(1.0, float(self.horizon))
        active_norm = active_count * self.coverage_norm_factor
        deactivation_norm = deactivation * self.coverage_norm_factor
        reward = (
            self.reward_alpha * norm_coverage
            + self.reward_beta * persistence_norm
            + self.reward_kappa * active_norm
            - (deactivation_norm * self.reward_eta)
            - saturation_penalty
        )
        return reward

    def _build_state(self) -> GraphState:
        t = min(self.current_t, self.dataset.num_steps - 1)
        flow_t = self.dataset.flows[t]
        flow_hist = self._history_tensor(flow_t)
        dyn_cap = self._current_dynamic_capacity()

        hist_mean = flow_hist.mean(dim=0)
        hist_std = flow_hist.std(dim=0, unbiased=False)
        raw_util = flow_t / torch.clamp(dyn_cap, min=1e-6)
        # Bound utilization to [0, 1) for the attention head while preserving
        # the ordering of over-capacity edges.
        util = raw_util / (1.0 + raw_util)
        same_comm = (
            self.dataset.community_ids[self.dataset.edge_index[0]]
            == self.dataset.community_ids[self.dataset.edge_index[1]]
        ).float()

        edge_attr_normalized = torch.stack(
            [
                flow_t / torch.clamp(self.dataset.flows.max(), min=1e-6),
                hist_mean / torch.clamp(self.dataset.flows.max(), min=1e-6),
                hist_std / torch.clamp(self.dataset.flows.max(), min=1e-6),
                self.static.capacity_norm,
                dyn_cap / torch.clamp(self.static.capacity.max(), min=1e-6),
                util,
                self.static.distance_norm,
                same_comm,
            ],
            dim=1,
        )

        trend_sma = torch.stack(list(self.utilization_history), dim=0).mean(dim=0)
        util_delta = trend_sma - self.prev_inbound_ratio
        dynamic_features = torch.stack(
            [
                self.active_mask,
                self.active_duration / float(self.horizon),
                self.prev_inbound_ratio,
                util_delta,
            ],
            dim=1,
        )
        structural_features = torch.stack(
            [
                self.static.indegree_norm,
                self.static.outdegree_norm,
                self.static.betweenness_norm,
            ],
            dim=1,
        )
        node_x = torch.cat([dynamic_features, structural_features], dim=1)

        return GraphState(
            x=node_x.to(self.device),
            edge_index=self.dataset.edge_index.to(self.device),
            edge_attr=edge_attr_normalized.to(self.device),
            global_t=t,
        )

    def step(self, action_idx: List[int]) -> tuple[GraphState, StepOutput]:
        if len(action_idx) > self.max_budget:
            action_idx = action_idx[: self.max_budget]

        flow_t = self.dataset.flows[self.current_t]
        dyn_cap = self._current_dynamic_capacity()

        seed_mask = torch.zeros(self.dataset.num_nodes, dtype=torch.float32)
        if action_idx:
            seed_mask[torch.tensor(action_idx, dtype=torch.long)] = 1.0

        source_active = torch.maximum(self.active_mask, seed_mask)

        src, dst = self.dataset.edge_index
        active_edge_flow = torch.minimum(flow_t, dyn_cap) * source_active[src]
        inbound = torch.zeros(self.dataset.num_nodes, dtype=torch.float32)
        inbound.index_add_(0, dst, active_edge_flow)

        has_inbound = (self.static.inbound_capacity > 0).float()
        next_active = ((inbound >= self.static.thresholds) * has_inbound).float()
        next_active = torch.maximum(next_active, seed_mask)

        prev_active_mask = self.active_mask.clone()
        coverage = float(((next_active > 0) & (self.visited_mask == 0)).sum().item())
        persistence = float(((prev_active_mask > 0) & (next_active > 0)).sum().item())
        deactivated_mask = (prev_active_mask > 0) & (next_active == 0)
        deactivation = float(deactivated_mask.sum().item())
        active_count = float(next_active.sum().item())
        cost = float(seed_mask.sum().item())
        saturation_penalty = self._compute_saturation_penalty()
        deactivation_penalty = (deactivation * self.coverage_norm_factor) * self.reward_eta
        reward = self._compute_reward(
            coverage=coverage,
            persistence=persistence,
            deactivation=deactivation,
            saturation_penalty=saturation_penalty,
            active_count=active_count,
            cost=cost,
        )

        self.visited_mask = torch.maximum(self.visited_mask, next_active)
        persisted_duration = torch.where(
            next_active > 0,
            torch.where(prev_active_mask > 0, self.active_duration + 1.0, torch.ones_like(self.active_duration)),
            torch.zeros_like(self.active_duration),
        )
        self.active_mask = next_active
        self.active_duration = persisted_duration
        self.prev_inbound_ratio = inbound / torch.clamp(self.static.inbound_capacity, min=1e-6)
        self.utilization_history.append(self.prev_inbound_ratio.clone())

        incident_fraction = float((self.episode_capacity_scale[self.step_idx] < 0.999).float().mean().item())

        self.current_t += 1
        self.step_idx += 1
        done = self.step_idx >= self.horizon or self.current_t >= self.dataset.num_steps

        next_state = self._build_state()
        out = StepOutput(
            reward=reward,
            coverage=coverage,
            persistence=persistence,
            deactivation=deactivation,
            deactivation_penalty=deactivation_penalty,
            saturation_penalty=saturation_penalty,
            cost=cost,
            active_count=active_count,
            done=done,
            incident_fraction=incident_fraction,
        )
        return next_state, out
