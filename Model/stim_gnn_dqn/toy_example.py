from __future__ import annotations

import torch

from .model import greedy_prefix_action


def _show_dynamic_budget_example(nodes: list[str], max_budget: int) -> None:
    snapshots = {
        "snapshot_1": torch.tensor([1.6, 0.9, -0.1, -0.6]),
        "snapshot_2": torch.tensor([0.4, -0.2, -0.5, -0.8]),
        "snapshot_3": torch.tensor([-0.1, -0.2, -0.4, -1.0]),
    }

    print("Dynamic-budget policy example:")
    print(f"Maximum budget K_max = {max_budget}")

    for name, q_values in snapshots.items():
        action_idx, prefix_value = greedy_prefix_action(q_values, max_budget)
        ranked_idx = torch.sort(q_values, descending=True).indices.tolist()
        chosen_nodes = [nodes[i] for i in action_idx.tolist()]
        ranked_nodes = [nodes[i] for i in ranked_idx]

        print(f"\n{name}:")
        print("Node scores:", {nodes[i]: round(float(q_values[i]), 2) for i in range(len(nodes))})
        print("Ranked nodes:", ranked_nodes)
        print(
            "Chosen budget:",
            len(chosen_nodes),
            f"(best prefix value = {float(prefix_value):.2f})",
        )
        print("Chosen nodes:", chosen_nodes)


def _diffusion_step(
    flow_t: torch.Tensor,
    capacity_t: torch.Tensor,
    seeds: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    thresholds: torch.Tensor,
    inbound_capacity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    effective_flow = torch.minimum(flow_t, capacity_t) * seeds[src]
    inbound = torch.zeros_like(thresholds)
    inbound.index_add_(0, dst, effective_flow)

    has_inbound = (inbound_capacity > 0).float()
    next_active = ((inbound >= thresholds) * has_inbound).float()
    next_active = torch.maximum(next_active, seeds)
    return inbound, next_active


def _lifespan_penalty(duration: torch.Tensor, eta: float, decay_lambda: float) -> float:
    if duration.numel() == 0:
        return 0.0
    weight = torch.exp(-decay_lambda * duration.float())
    return float((eta * weight.sum()).item())


def _show_diffusion_example(nodes: list[str], node_to_idx: dict[str, int]) -> None:
    reward_alpha = 0.02
    reward_beta = 1.10
    reward_kappa = 0.05
    reward_eta = 1.20
    reward_deactivation_lambda = 0.10
    reward_saturation_threshold = 0.80
    reward_saturation_weight = 2.00
    reward_zeta = 0.05

    edges = [("A", "C"), ("B", "C"), ("C", "D")]
    src = torch.tensor([node_to_idx[u] for u, _ in edges], dtype=torch.long)
    dst = torch.tensor([node_to_idx[v] for _, v in edges], dtype=torch.long)

    c_static = torch.tensor([8.0, 10.0, 7.0])
    inbound_capacity = torch.zeros(len(nodes))
    inbound_capacity.index_add_(0, dst, c_static)

    alpha_th = 0.5
    theta = alpha_th * inbound_capacity

    print("\nCapacity-constrained diffusion example:")
    print("Static capacities C:", c_static.tolist())
    print("Thresholds theta:", theta.tolist())

    flow_t = torch.tensor([5.0, 12.0, 6.0])
    seeds = torch.tensor([0.0, 1.0, 0.0, 0.0])  # seed B

    inbound, next_active = _diffusion_step(
        flow_t=flow_t,
        capacity_t=c_static,
        seeds=seeds,
        src=src,
        dst=dst,
        thresholds=theta,
        inbound_capacity=inbound_capacity,
    )

    print("\nNormal conditions:")
    print("Chosen seed:", [nodes[i] for i in torch.where(seeds > 0)[0].tolist()])
    print("Effective inbound flow:", inbound.tolist())
    print("Next active:", {n: int(next_active[i].item()) for i, n in enumerate(nodes)})

    rho = torch.tensor([1.0, 0.6, 1.0])
    c_storm = rho * c_static
    inbound_storm, next_active_storm = _diffusion_step(
        flow_t=flow_t,
        capacity_t=c_storm,
        seeds=seeds,
        src=src,
        dst=dst,
        thresholds=theta,
        inbound_capacity=inbound_capacity,
    )

    print("\nStorm conditions (dynamic C_t = rho * C_static):")
    print("Dynamic capacities:", c_storm.tolist())
    print("Effective inbound flow:", inbound_storm.tolist())
    print("Next active:", {n: int(next_active_storm[i].item()) for i, n in enumerate(nodes)})

    coverage = float(((next_active > 0) & (torch.zeros_like(next_active) == 0)).sum().item())
    persistence = 0.0
    deactivation = 0.0
    active_count = float(next_active.sum().item())
    cost = float(seeds.sum().item())
    deactivation_penalty = 0.0
    saturation = inbound / torch.clamp(theta, min=1e-6)
    saturation_penalty = float(
        torch.where(
            (theta > 0) & (saturation > reward_saturation_threshold),
            (saturation - reward_saturation_threshold) * reward_saturation_weight,
            torch.zeros_like(saturation),
        ).sum().item()
    )
    reward = (
        reward_alpha * coverage
        + reward_beta * persistence
        + reward_kappa * active_count
        - deactivation_penalty
        - saturation_penalty
        - reward_zeta * cost
    )

    print("\nReward example under the reliability-oriented objective:")
    print(
        f"coverage={coverage:.1f}, persistence={persistence:.1f}, "
        f"active_count={active_count:.1f}, deactivation={deactivation:.1f}, "
        f"deactivation_penalty={deactivation_penalty:.2f}, saturation_penalty={saturation_penalty:.2f}, "
        f"cost={cost:.1f}, reward={reward:.2f}"
    )

    prev_active = torch.tensor([0.0, 1.0, 1.0, 0.0])
    prev_duration = torch.tensor([0.0, 3.0, 1.0, 0.0])
    persistence_storm = float(((prev_active > 0) & (next_active_storm > 0)).sum().item())
    deactivation_storm = float(((prev_active > 0) & (next_active_storm == 0)).sum().item())
    coverage_storm = float(((next_active_storm > 0) & (prev_active == 0)).sum().item())
    active_count_storm = float(next_active_storm.sum().item())
    storm_deactivated = (prev_active > 0) & (next_active_storm == 0)
    deactivation_penalty_storm = _lifespan_penalty(
        prev_duration[storm_deactivated], reward_eta, reward_deactivation_lambda
    )
    saturation_storm = inbound_storm / torch.clamp(theta, min=1e-6)
    saturation_penalty_storm = float(
        torch.where(
            (theta > 0) & (saturation_storm > reward_saturation_threshold),
            (saturation_storm - reward_saturation_threshold) * reward_saturation_weight,
            torch.zeros_like(saturation_storm),
        ).sum().item()
    )
    reward_storm = (
        reward_alpha * coverage_storm
        + reward_beta * persistence_storm
        + reward_kappa * active_count_storm
        - deactivation_penalty_storm
        - saturation_penalty_storm
        - reward_zeta * cost
    )

    print("\nReliability penalty example after a dropout:")
    print("Previous active:", {n: int(prev_active[i].item()) for i, n in enumerate(nodes)})
    print("Previous durations:", {n: float(prev_duration[i].item()) for i, n in enumerate(nodes)})
    print("Storm next active:", {n: int(next_active_storm[i].item()) for i, n in enumerate(nodes)})
    print(
        f"coverage={coverage_storm:.1f}, persistence={persistence_storm:.1f}, "
        f"active_count={active_count_storm:.1f}, deactivation={deactivation_storm:.1f}, "
        f"deactivation_penalty={deactivation_penalty_storm:.2f}, saturation_penalty={saturation_penalty_storm:.2f}, "
        f"cost={cost:.1f}, reward={reward_storm:.2f}"
    )


def _show_baseline_reward_example(nodes: list[str], node_to_idx: dict[str, int]) -> None:
    reward_alpha = 0.02
    reward_beta = 1.10
    reward_kappa = 0.05
    reward_eta = 1.20
    reward_deactivation_lambda = 0.10
    reward_saturation_threshold = 0.80
    reward_saturation_weight = 2.00
    reward_zeta = 0.05

    edges = [("A", "C"), ("B", "C"), ("C", "D")]
    src = torch.tensor([node_to_idx[u] for u, _ in edges], dtype=torch.long)
    dst = torch.tensor([node_to_idx[v] for _, v in edges], dtype=torch.long)

    c_static = torch.tensor([8.0, 10.0, 7.0])
    inbound_capacity = torch.zeros(len(nodes))
    inbound_capacity.index_add_(0, dst, c_static)
    theta = 0.5 * inbound_capacity
    flow_t = torch.tensor([5.0, 12.0, 6.0])

    eff = torch.minimum(flow_t, c_static)
    greedy_score = torch.zeros(len(nodes))
    greedy_score.index_add_(0, src, eff / torch.clamp(theta[dst], min=1e-6))

    chosen_idx = int(torch.argmax(greedy_score).item())
    seeds = torch.zeros(len(nodes))
    seeds[chosen_idx] = 1.0
    inbound, next_active = _diffusion_step(
        flow_t=flow_t,
        capacity_t=c_static,
        seeds=seeds,
        src=src,
        dst=dst,
        thresholds=theta,
        inbound_capacity=inbound_capacity,
    )

    coverage = float(((next_active > 0) & (torch.zeros_like(next_active) == 0)).sum().item())
    persistence = 0.0
    deactivation = 0.0
    active_count = float(next_active.sum().item())
    cost = float(seeds.sum().item())
    deactivation_penalty = _lifespan_penalty(torch.empty(0), reward_eta, reward_deactivation_lambda)
    saturation = inbound / torch.clamp(theta, min=1e-6)
    saturation_penalty = float(
        torch.where(
            (theta > 0) & (saturation > reward_saturation_threshold),
            (saturation - reward_saturation_threshold) * reward_saturation_weight,
            torch.zeros_like(saturation),
        ).sum().item()
    )
    reward = (
        reward_alpha * coverage
        + reward_beta * persistence
        + reward_kappa * active_count
        - deactivation_penalty
        - saturation_penalty
        - reward_zeta * cost
    )

    print("\nHow a baseline uses the reward:")
    print("Greedy one-step scores:", {nodes[i]: round(float(greedy_score[i]), 2) for i in range(len(nodes))})
    print("Greedy chooses:", nodes[chosen_idx], "(from the heuristic score, not from reward directly)")
    print("Environment then computes the reward for that choice:")
    print(
        f"coverage={coverage:.1f}, persistence={persistence:.1f}, "
        f"active_count={active_count:.1f}, deactivation={deactivation:.1f}, "
        f"deactivation_penalty={deactivation_penalty:.2f}, saturation_penalty={saturation_penalty:.2f}, "
        f"cost={cost:.1f}, reward={reward:.2f}"
    )
    print("Effective inbound flow after the greedy action:", inbound.tolist())


def main() -> None:
    nodes = ["A", "B", "C", "D"]
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    _show_dynamic_budget_example(nodes, max_budget=3)
    _show_diffusion_example(nodes, node_to_idx)
    _show_baseline_reward_example(nodes, node_to_idx)


if __name__ == "__main__":
    main()
