from __future__ import annotations

import copy
import json
import random

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .data_loader import build_dataset
from .environment import CapacityConstrainedEnv
from .incidents import (
    build_incident_schedule_for_episode,
    map_incidents_to_segments,
    write_incident_mapping_reports,
)
from .model import GNNQNetwork, action_value, greedy_prefix_action, select_action
from .preprocess import build_static_parameters
from .replay_buffer import ReplayBuffer, Transition


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def epsilon_by_step(step: int, cfg: Config) -> float:
    if step >= cfg.epsilon_decay_steps:
        return cfg.epsilon_end
    frac = step / max(1, cfg.epsilon_decay_steps)
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def optimize_model(
    q_net: GNNQNetwork,
    target_net: GNNQNetwork,
    buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    device: str,
) -> float:
    if len(buffer) < cfg.batch_size:
        return 0.0

    batch = buffer.sample(cfg.batch_size)
    losses = []

    for tr in batch:
        q_values = q_net(tr.state)
        q_action = action_value(q_values, tr.action_idx.to(device))

        with torch.no_grad():
            next_q_values = target_net(tr.next_state)
            next_action, _ = greedy_prefix_action(next_q_values, cfg.max_budget)
            next_q = action_value(next_q_values, next_action)
            target = torch.tensor(tr.reward, dtype=torch.float32, device=device)
            if not tr.done:
                target = target + cfg.gamma * next_q

        losses.append(F.mse_loss(q_action, target))

    optimizer.zero_grad()
    loss = torch.stack(losses).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), cfg.max_grad_norm)
    optimizer.step()
    return float(loss.item())


def build_optimizer(q_net: GNNQNetwork, cfg: Config) -> torch.optim.Optimizer:
    return torch.optim.Adam(q_net.parameters(), lr=cfg.lr)


def soft_update_target(target_net: GNNQNetwork, q_net: GNNQNetwork, tau: float) -> None:
    with torch.no_grad():
        for target_param, q_param in zip(target_net.parameters(), q_net.parameters()):
            target_param.data.copy_(tau * q_param.data + (1.0 - tau) * target_param.data)


def _write_split_summary(cfg: Config, dataset, static, out_dir) -> None:
    summary = {
        "num_steps": int(dataset.num_steps),
        "horizon": int(cfg.horizon),
        "train_days": int(len(static.train_starts)),
        "val_days": int(len(static.val_starts)),
        "test_days": int(len(static.test_starts)),
        "validation_start_config": cfg.validation_start,
        "validation_end_config": cfg.validation_end,
    }
    if dataset.timestamps is not None:
        ts = dataset.timestamps

        def _window(starts: list[int]) -> dict[str, str | int] | None:
            if not starts:
                return None
            return {
                "start_index": int(starts[0]),
                "end_index": int(starts[-1] + cfg.horizon - 1),
                "start_time": str(ts[starts[0]]),
                "end_time": str(ts[starts[-1] + cfg.horizon - 1]),
            }

        summary["train_window"] = _window(static.train_starts)
        summary["val_window"] = _window(static.val_starts)
        summary["test_window"] = _window(static.test_starts)
    (out_dir / "data_split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _run_greedy_episode_reward(
    env: CapacityConstrainedEnv,
    q_net: GNNQNetwork,
    start_t: int,
    device: str,
    incident_schedule: torch.FloatTensor | None = None,
) -> float:
    state = env.reset(start_t, incident_schedule=incident_schedule)
    done = False
    total_reward = 0.0
    while not done:
        with torch.no_grad():
            action_idx = select_action(q_net, state, env.max_budget, epsilon=0.0, device=device)
        next_state, out = env.step(action_idx.detach().cpu().tolist())
        total_reward += out.reward
        state = next_state
        done = out.done
    return float(total_reward)


def _validation_reward(
    cfg: Config,
    q_net: GNNQNetwork,
    val_env: CapacityConstrainedEnv,
    static,
    dataset,
    mapped_incidents,
    device: str,
) -> float:
    if not static.val_starts:
        return float("-inf")

    was_training = q_net.training
    q_net.eval()

    starts = static.val_starts[: max(1, min(cfg.val_eval_episodes, len(static.val_starts)))]
    rewards = []
    for start_t in starts:
        if mapped_incidents is not None and dataset.timestamps is not None:
            incident_schedule = build_incident_schedule_for_episode(
                start_t=start_t,
                horizon=cfg.horizon,
                num_edges=dataset.num_edges,
                timestamps=dataset.timestamps,
                mapped_incidents=mapped_incidents,
            )
        else:
            incident_schedule = val_env.sample_incident_schedule()
        rewards.append(
            _run_greedy_episode_reward(
                env=val_env,
                q_net=q_net,
                start_t=start_t,
                device=device,
                incident_schedule=incident_schedule,
            )
        )

    if was_training:
        q_net.train()
    return float(np.mean(rewards)) if rewards else float("-inf")


def train_rl(
    cfg: Config | None = None,
    save_artifacts: bool = True,
) -> tuple[GNNQNetwork, list[tuple[float, float, float, float, float, float]]]:
    cfg = cfg or Config()
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
    out_dir = cfg.workdir_path
    if save_artifacts:
        _write_split_summary(cfg, dataset, static, out_dir)
    mapped_incidents = None

    if (
        dataset.timestamps is not None
        and cfg.incident_events_xlsx is not None
        and cfg.incident_metadata_csv is not None
    ):
        incident_map = map_incidents_to_segments(
            incidents_xlsx=cfg.incident_events_xlsx,
            metadata_csv=cfg.incident_metadata_csv,
            both_factor=cfg.incident_both_directions_factor,
            single_factor=cfg.incident_single_direction_factor,
        )
        if save_artifacts:
            report_paths = write_incident_mapping_reports(
                mapping=incident_map,
                metadata_csv=cfg.incident_metadata_csv,
                output_dir=out_dir / "incident_reports_train",
            )
            print(
                "Incident mapping summary: "
                f"mapped_rows={incident_map.mapped_rows} unmapped_rows={incident_map.unmapped_rows}"
            )
            print(f"Incident mapping reports saved to {report_paths['summary_json']}")
        mapped_incidents = incident_map.mapped

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
        incident_prob=cfg.incident_prob_train,
        incident_factor_low=cfg.incident_factor_low,
        incident_factor_high=cfg.incident_factor_high,
        incident_duration_min=cfg.incident_duration_min,
        incident_duration_max=cfg.incident_duration_max,
        incident_edge_fraction=cfg.incident_edge_fraction,
        device=device,
    )
    val_env = CapacityConstrainedEnv(
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

    sample_state = env.reset(static.train_starts[0])
    q_net = GNNQNetwork(
        node_in_dim=sample_state.x.shape[1],
        edge_in_dim=sample_state.edge_attr.shape[1],
        hidden_dim=cfg.hidden_dim,
        gnn_layers=cfg.gnn_layers,
    ).to(device)
    target_net = copy.deepcopy(q_net)
    optimizer = build_optimizer(q_net, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, cfg.episodes),
        eta_min=cfg.lr_min,
    )
    buffer = ReplayBuffer(cfg.replay_capacity)

    global_step = 0
    history = []
    best_val_reward = -float("inf")
    best_episode = -1
    best_model_path = out_dir / "best_q_net.pt"
    has_optimizer_step = False
    best_state_dict = None

    for episode in range(cfg.episodes):
        start_t = random.choice(static.train_starts)
        state = env.reset(start_t)
        done = False
        episode_reward = 0.0
        episode_loss = 0.0
        episode_budget = 0.0
        steps = 0

        while not done:
            epsilon = epsilon_by_step(global_step, cfg)
            action_idx = select_action(q_net, state, cfg.max_budget, epsilon, device=device)
            next_state, out = env.step(action_idx.detach().cpu().tolist())

            buffer.push(
                Transition(
                    state=state,
                    action_idx=action_idx.detach().cpu(),
                    reward=out.reward,
                    next_state=next_state,
                    done=out.done,
                )
            )

            if len(buffer) >= cfg.warmup_steps:
                step_loss = optimize_model(q_net, target_net, buffer, optimizer, cfg, device)
                episode_loss += step_loss
                if step_loss > 0.0:
                    has_optimizer_step = True

            soft_update_target(target_net, q_net, cfg.target_update_tau)

            state = next_state
            episode_reward += out.reward
            episode_budget += out.cost
            steps += 1
            global_step += 1
            done = out.done

        avg_loss = episode_loss / max(1, steps)
        avg_budget = episode_budget / max(1, steps)
        val_reward = float("nan")
        if episode % 5 == 0:
            val_metrics = _validation_reward(
                cfg=cfg,
                q_net=q_net,
                val_env=val_env,
                static=static,
                dataset=dataset,
                mapped_incidents=mapped_incidents,
                device=device,
            )
            val_reward = val_metrics
            if val_metrics > best_val_reward:
                best_val_reward = val_metrics
                best_episode = episode
                best_state_dict = copy.deepcopy(q_net.state_dict())
                if save_artifacts:
                    torch.save(q_net.state_dict(), best_model_path)
                print(f"New best model saved at episode {episode}")

        if has_optimizer_step:
            scheduler.step()
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append((episode, episode_reward, avg_loss, avg_budget, val_reward, current_lr))
        print(
            f"episode={episode:03d} reward={episode_reward:8.3f} "
            f"avg_loss={avg_loss:8.5f} avg_budget={avg_budget:6.3f} "
            f"val_reward={val_reward:8.3f} lr={current_lr:.6f}"
        )

    # Return the best model found on validation when available.
    if best_state_dict is not None:
        q_net.load_state_dict(best_state_dict)

    if save_artifacts:
        torch.save(q_net.state_dict(), out_dir / "q_net.pt")
        np.savetxt(
            out_dir / "training_history.csv",
            np.array(history),
            delimiter=",",
            header="episode,reward,avg_loss,avg_budget,val_reward,lr",
            comments="",
        )
        (out_dir / "best_model_summary.json").write_text(
            json.dumps(
                {
                    "best_episode": int(best_episode),
                    "best_val_reward": float(best_val_reward),
                    "best_model_path": str(best_model_path),
                    "final_model_path": str(out_dir / "q_net.pt"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if best_model_path.exists():
            print(
                f"Saved best model to {best_model_path} "
                f"(episode={best_episode:03d}, val_reward={best_val_reward:8.3f})"
            )
        print(f"Saved model to {out_dir / 'q_net.pt'}")
        print(f"Saved history to {out_dir / 'training_history.csv'}")
    return q_net, history


def main() -> None:
    train_rl(Config(), save_artifacts=True)


if __name__ == "__main__":
    main()
