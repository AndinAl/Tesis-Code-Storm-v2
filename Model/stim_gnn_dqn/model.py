from __future__ import annotations

import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .environment import GraphState


class AttentionMessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        msg_input = torch.cat([h[src], edge_attr], dim=1)
        msg = self.msg_mlp(msg_input)

        attn_input = torch.cat([h[src], h[dst], edge_attr], dim=1)
        scores = self.attn_mlp(attn_input).squeeze(-1)

        exp_scores = torch.exp(scores - scores.max())
        sum_exp = torch.zeros(h.shape[0], device=h.device)
        sum_exp.index_add_(0, dst, exp_scores)
        weights = exp_scores / (sum_exp[dst] + 1e-6)

        weighted_msg = msg * weights.unsqueeze(1)
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, weighted_msg)

        out = self.upd_mlp(torch.cat([h, agg], dim=1))
        return self.norm(F.relu(out + h))


class GNNQNetwork(nn.Module):
    def __init__(self, node_in_dim: int, edge_in_dim: int, hidden_dim: int, gnn_layers: int) -> None:
        super().__init__()
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [AttentionMessagePassingLayer(hidden_dim, edge_in_dim) for _ in range(gnn_layers)]
        )
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: GraphState) -> torch.Tensor:
        h = F.relu(self.node_proj(state.x))
        for layer in self.layers:
            h = layer(h, state.edge_index, state.edge_attr)
        return self.q_head(h).squeeze(-1)


def greedy_prefix_action(q_values: torch.Tensor, max_budget: int) -> Tuple[torch.LongTensor, torch.Tensor]:
    n = q_values.numel()
    limit = min(max_budget, n)
    if limit <= 0:
        empty = torch.empty(0, dtype=torch.long, device=q_values.device)
        return empty, torch.tensor(0.0, dtype=q_values.dtype, device=q_values.device)

    sorted_q, sorted_idx = torch.sort(q_values, descending=True)
    prefix_values = torch.cumsum(sorted_q[:limit], dim=0)
    candidate_values = torch.cat(
        [torch.zeros(1, dtype=q_values.dtype, device=q_values.device), prefix_values],
        dim=0,
    )
    best_budget = int(torch.argmax(candidate_values).item())
    return sorted_idx[:best_budget], candidate_values[best_budget]


def select_action(
    q_net: GNNQNetwork,
    state: GraphState,
    max_budget: int,
    epsilon: float,
    device: str = "cpu",
) -> torch.LongTensor:
    n = state.x.shape[0]
    if random.random() < epsilon:
        budget = random.randint(0, min(max_budget, n))
        choice = random.sample(range(n), k=budget)
        return torch.tensor(choice, dtype=torch.long, device=device)

    with torch.no_grad():
        q_values = q_net(state)
        action_idx, _ = greedy_prefix_action(q_values, max_budget)
        return action_idx


def action_value(q_values: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
    if action_idx.numel() == 0:
        return torch.tensor(0.0, dtype=q_values.dtype, device=q_values.device)
    return q_values[action_idx].sum()
