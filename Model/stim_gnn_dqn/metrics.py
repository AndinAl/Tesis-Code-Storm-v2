from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class EpisodeMetrics:
    cumulative_reward: float
    j_int: float
    stability_score: float
    mean_time_to_deactivation: float
    flicker_rate: float
    avg_deactivation_count: float
    avg_active_count: float
    avg_budget_used: float
    total_budget_used: float
    avg_incident_fraction: float


def summarize_episode(
    rewards: List[float],
    active_counts: List[float],
    active_masks: List[np.ndarray],
    actions: List[List[int]],
    deactivation_counts: List[float],
    costs: List[float],
    incident_fracs: List[float],
) -> EpisodeMetrics:
    if not active_masks:
        return EpisodeMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    active_mat = np.stack(active_masks, axis=0)
    deactivations = (active_mat[:-1] > active_mat[1:]).sum(axis=0)
    flicker_rate = float(deactivations.mean())

    avg_active_count = float(np.mean(active_counts))
    stability_score = float(np.mean(active_counts) / max(1.0, active_mat.shape[1]))
    j_int = float(np.sum(active_counts))

    lifetimes = []
    for t, action in enumerate(actions):
        for node_idx in action:
            duration = 0
            for s in range(t, active_mat.shape[0]):
                if active_mat[s, node_idx] > 0:
                    duration += 1
                else:
                    break
            lifetimes.append(duration)

    return EpisodeMetrics(
        cumulative_reward=float(np.sum(rewards)),
        j_int=j_int,
        stability_score=stability_score,
        mean_time_to_deactivation=float(np.mean(lifetimes)) if lifetimes else 0.0,
        flicker_rate=flicker_rate,
        avg_deactivation_count=float(np.mean(deactivation_counts)) if deactivation_counts else 0.0,
        avg_active_count=avg_active_count,
        avg_budget_used=float(np.mean(costs)) if costs else 0.0,
        total_budget_used=float(np.sum(costs)) if costs else 0.0,
        avg_incident_fraction=float(np.mean(incident_fracs)) if incident_fracs else 0.0,
    )
