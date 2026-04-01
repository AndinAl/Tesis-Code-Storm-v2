# Tesis Code Storm v2

Traffic-flow imputation + directional RL control pipeline for the RS road network.

This project has two connected parts:

1. `flow_imputation`: builds dense hourly directional edge flow time series from sparse sensor data.
2. `stim_gnn_dqn`: trains and evaluates a GNN-DQN agent for capacity-constrained influence/intervention decisions on the directed road graph.

## Repository structure

```text
Data/
  data/
    network.json
    pntc_rs_unificado.parquet
    ocorrencias-interdicoes_rs_chuvas_2024.xlsx
Model/
  flow_imputation/
  stim_gnn_dqn/
  outputs_imputation/
  outputs/
```

## Main inputs

- `Data/data/network.json`: directed road graph source.
- `Data/data/pntc_rs_unificado.parquet`: observed hourly traffic counts.
- `Model/outputs_imputation/imputed_edge_flow_matrix.npz`: imputed directional flow matrix (`T x E`).
- `Model/outputs_imputation/imputed_edge_metadata.csv`: metadata aligned to imputed edge columns.
- `Data/data/ocorrencias-interdicoes_rs_chuvas_2024.xlsx`: incidents/interdictions used for incident-aware evaluation.

## Environment setup

From repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Model/requirements.txt
```

## Train flow imputer

```bash
cd Model
python -m flow_imputation.train_imputer
```

Outputs are written to `Model/outputs_imputation/`.

## Train RL model (stim_gnn_dqn)

```bash
cd Model
python -m stim_gnn_dqn.train
```

Default config is in `Model/stim_gnn_dqn/config.py`. Current defaults include:

- directional edge guardrail: `expected_directed_edges = 852`
- `max_budget = 10`
- `episodes = 180`
- `lr = 1e-4`, `lr_min = 1e-5`
- `batch_size = 64`
- `epsilon_decay_steps = 5000`

## Evaluate RL vs baselines

```bash
cd Model
python -m stim_gnn_dqn.evaluate
```

Evaluation compares:

- `random`
- `greedy`
- `degree`
- `rl`
- `storm` (stress test)

## Persistent training with tmux

Start a persistent session:

```bash
tmux new -s rl_train
```

Inside tmux:

```bash
cd "/home/andina/Tesis_storm_v2/Tesis Code Storm v2/Model"
source ../.venv/bin/activate
python -m stim_gnn_dqn.train | tee outputs/train_$(date +%Y%m%d_%H%M%S).log
```

Detach and leave training running:

- Press `Ctrl+b`, then `d`

Later:

```bash
tmux ls
tmux attach -t rl_train
```

## Notes

- The train/validation split excludes the configured flood window:
  `2024-04-30 02:30:00` to `2024-06-22 19:14:00`.
- Incident mapping reports are exported under:
  - `Model/outputs/incident_reports_train/`
  - `Model/outputs/incident_reports_eval/`
- For implementation details and module-level notes, see `Model/README.md`.

