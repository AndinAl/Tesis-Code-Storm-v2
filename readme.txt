Tesis Code Storm v2 - STIM GNN-DQN Technical Notes

Date: 2026-04-04

1) Environment and requirements

Create environment:
  python -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r Model/requirements.txt

Current requirements (Model/requirements.txt):
  numpy>=1.24
  pandas>=2.0
  openpyxl>=3.1
  pyarrow>=10.0
  torch>=2.2
  torch-geometric>=2.7
  networkx>=3.0
  matplotlib>=3.7

2) Data and preprocessing

Main data inputs:
  Data/data/network.json
  Data/data/pntc_rs_unificado.parquet
  Data/data/ocorrencias-interdicoes_rs_chuvas_2024.xlsx

Optional imputation outputs used by the RL loader:
  Model/outputs_imputation/imputed_edge_flow_matrix.npz
  Model/outputs_imputation/imputed_edge_metadata.csv

Data loader behavior:
  - Preferred mode: load directional flow matrix from NPZ (source, target, direction, flow).
  - Graph guardrail: expected_directed_edges=852.
  - Fallback mode: legacy graph_dict + XLSX files.

Preprocessing (preprocess.py):
  - Builds static parameters from train split:
    capacity, thresholds, inbound/outbound capacities, normalized centralities.
  - Uses:
    capacity_quantile=0.95
    threshold_alpha=0.66

3) Model (stim_gnn_dqn)

Core files:
  Model/stim_gnn_dqn/model.py
  Model/stim_gnn_dqn/environment.py
  Model/stim_gnn_dqn/train.py
  Model/stim_gnn_dqn/evaluate.py

Architecture summary:
  - Node/edge graph state with directional edges.
  - Attention message passing GNN for node-level Q values.
  - DQN-style optimization with replay buffer + target network soft update.
  - Capacity-constrained intervention policy (max_budget).

Reward structure (environment):
  reward = alpha*coverage + beta*persistence + kappa*active_count
           - eta*deactivation - saturation_penalty
  Coverage and deactivation are normalized by number of nodes.

4) Default training parameters (config.py)

Temporal/data:
  horizon=24
  history_len=4
  train_ratio=0.70
  val_ratio=0.15
  validation_start=2024-04-30 02:30:00
  validation_end=2024-06-22 19:14:00

Reward:
  reward_alpha=0.01
  reward_beta=5.00
  reward_kappa=0.15
  reward_eta=5.00
  reward_deactivation_lambda=0.10
  reward_saturation_threshold=0.72
  reward_saturation_weight=5.00

Model/action:
  max_budget=10
  hidden_dim=64
  gnn_layers=3

Optimization:
  episodes=180
  lr=1e-4
  lr_min=1e-5
  batch_size=64
  replay_capacity=5000
  warmup_steps=64
  target_update_tau=0.005
  max_grad_norm=1.0
  epsilon_decay_steps=5000

5) Baselines used in evaluation

Policies:
  random
  greedy
  degree
  rl
  storm (stress scenario)

Main reported metrics:
  J (cumulative reward)
  J_int (influence volume / active count accumulation)
  MTTD (mean time to deactivation)
  phi (flicker rate)
  D (avg deactivation count)
  B (avg budget used)

6) Run commands

Train:
  cd Model
  python -m stim_gnn_dqn.train

Evaluate:
  cd Model
  python -m stim_gnn_dqn.evaluate

Generate paper/report artifacts:
  cd Model
  python -m stim_gnn_dqn.report_artifacts

7) Expected results

Expected outputs after train:
  Model/outputs/q_net.pt
  Model/outputs/best_q_net.pt (if validation model selection saves best)
  Model/outputs/training_history.csv
  Model/outputs/data_split_summary.json
  Model/outputs/incident_reports_train/*

Expected outputs after report generation:
  Model/outputs/paper_assets/*.pdf
  Model/outputs/paper_assets/analysis_tables.tex.txt
  Model/outputs/paper_assets/report_summary.txt

Expected behavior:
  - RL should outperform random on J and J_int.
  - RL should be competitive with greedy/degree while controlling flicker (phi).
  - Incident-aware runs should show directional response under stress scenarios.
