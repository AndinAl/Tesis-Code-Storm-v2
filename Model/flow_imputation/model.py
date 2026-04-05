from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import ChebConv

try:
    from torch_geometric_temporal.nn.recurrent import GConvGRU as TemporalGConvGRU
except ImportError:
    TemporalGConvGRU = None


class _FallbackGConvGRU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K: int = 2) -> None:
        super().__init__()
        gate_channels = in_channels + out_channels
        self.out_channels = out_channels
        self.conv_z = ChebConv(gate_channels, out_channels, K=K)
        self.conv_r = ChebConv(gate_channels, out_channels, K=K)
        self.conv_h = ChebConv(gate_channels, out_channels, K=K)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_state is None:
            hidden_state = x.new_zeros((x.size(0), self.out_channels))

        gate_input = torch.cat([x, hidden_state], dim=-1)
        update_gate = torch.sigmoid(self.conv_z(gate_input, edge_index, edge_weight))
        reset_gate = torch.sigmoid(self.conv_r(gate_input, edge_index, edge_weight))
        candidate_input = torch.cat([x, reset_gate * hidden_state], dim=-1)
        candidate = torch.tanh(self.conv_h(candidate_input, edge_index, edge_weight))
        return (1.0 - update_gate) * hidden_state + update_gate * candidate


class STGNNImputer(nn.Module):
    def __init__(self, node_features: int, hidden_dim: int, K: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        if TemporalGConvGRU is not None:
            self.recurrent: nn.Module = TemporalGConvGRU(node_features, hidden_dim, K=K)
        else:
            self.recurrent = _FallbackGConvGRU(node_features, hidden_dim, K=K)
        self.dropout = nn.Dropout(dropout)
        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.recurrent(x, edge_index, edge_weight, hidden_state)
        hidden = self.dropout(F.relu(hidden))
        prediction = self.output_head(hidden).squeeze(-1)
        return prediction, hidden
