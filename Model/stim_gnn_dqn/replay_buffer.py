from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List

import torch

from .environment import GraphState


@dataclass
class Transition:
    state: GraphState
    action_idx: torch.LongTensor
    reward: float
    next_state: GraphState
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.data: Deque[Transition] = deque(maxlen=capacity)

    def push(self, tr: Transition) -> None:
        self.data.append(
            Transition(
                state=tr.state.clone(),
                action_idx=tr.action_idx.clone().detach(),
                reward=float(tr.reward),
                next_state=tr.next_state.clone(),
                done=bool(tr.done),
            )
        )

    def sample(self, batch_size: int) -> List[Transition]:
        batch_size = min(batch_size, len(self.data))
        return random.sample(list(self.data), batch_size)

    def __len__(self) -> int:
        return len(self.data)
