Tesis Code Storm v2 - Archive and Report Notes

Date: 2026-04-04

Latest kept archive:
- Model/outputs/paper_assets/stim_gnn_dqn_code_2026-04-01.zip

Removed archives:
- Model/outputs/paper_assets/best_validation_figures_2026-04-01.zip
- Model/outputs/paper_assets/best_validation_summary_2026-04-01.zip

Added report file:
- Model/outputs/paper_assets/report_last_run_parameters_2026-04-04.txt

Quick run:
1) python -m venv .venv
2) source .venv/bin/activate
3) python -m pip install -r Model/requirements.txt
4) cd Model && python -m stim_gnn_dqn.train
5) python -m stim_gnn_dqn.report_artifacts
