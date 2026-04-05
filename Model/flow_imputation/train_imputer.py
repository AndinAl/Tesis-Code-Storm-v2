from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd
import torch

from .imputation import (
    ImputationConfig,
    export_imputed_flow_matrix,
    export_segment_metadata,
    impute_full_flow_matrix,
    prepare_imputation_dataset,
    set_seed,
    train_imputer,
)


def main() -> None:
    cfg = ImputationConfig()
    set_seed(cfg.seed)

    dataset = prepare_imputation_dataset(cfg)
    print(
        f"steps={dataset.num_steps} directed_segments={dataset.num_segments} "
        f"matched_sensor_pairs={dataset.matched_sensor_pairs} "
        f"unmatched_sensor_pairs={len(dataset.unmatched_sensor_pairs)}"
    )

    model, history = train_imputer(dataset, cfg)

    out_dir = cfg.workdir_path
    torch.save(model.state_dict(), out_dir / cfg.model_filename)
    pd.DataFrame([asdict(metric) for metric in history]).to_csv(
        out_dir / cfg.history_filename,
        index=False,
    )

    completed_flow = impute_full_flow_matrix(model, dataset)
    export_imputed_flow_matrix(dataset, completed_flow, out_dir / cfg.completed_flow_filename)
    export_segment_metadata(dataset, out_dir / cfg.metadata_filename)
    (out_dir / "imputer_config.json").write_text(
        json.dumps(asdict(cfg), indent=2),
        encoding="utf-8",
    )

    print(f"Saved model to {out_dir / cfg.model_filename}")
    print(f"Saved training history to {out_dir / cfg.history_filename}")
    print(f"Saved imputed flow matrix to {out_dir / cfg.completed_flow_filename}")
    print(f"Saved segment metadata to {out_dir / cfg.metadata_filename}")


if __name__ == "__main__":
    main()
