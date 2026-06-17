from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_rote_models_round2 import (
    SplitIndex,
    load_npz,
    set_seed,
    train_one_model,
)
from train_collected_v3 import eval_one_model


EXPERIMENTS = [
    {
        "model": "TSMixer-Raw",
        "feature_indices": [0],
        "description": "raw_latency only",
    },
    {
        "model": "TSMixer-BasicStat",
        "feature_indices": [0, 1, 2, 3],
        "description": "raw_latency, delta_latency, rolling_mean, rolling_std",
    },
    {
        "model": "TSMixer-OPR-v2",
        "feature_indices": None,
        "description": "full OPR-v2 risk-onset representation",
    },
]


def select_features(data: dict, indices: list[int] | None) -> dict:
    selected = dict(data)
    x = data["X_opr"].astype(np.float32)
    selected["X_opr"] = x if indices is None else x[:, :, indices]
    return selected


def make_combined(train: dict, val: dict) -> dict:
    return {
        "X_opr": np.concatenate([train["X_opr"], val["X_opr"]], axis=0),
        "y_risk": np.concatenate([train["y_risk"], val["y_risk"]], axis=0),
        "time_to_risk": np.concatenate([train["time_to_risk"], val["time_to_risk"]], axis=0),
        "risk_intensity": np.concatenate([train["risk_intensity"], val["risk_intensity"]], axis=0),
        "state_cls": np.concatenate([train["state_cls"], val["state_cls"]], axis=0),
        "current_normal": np.concatenate([train["current_normal"], val["current_normal"]], axis=0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train TSMixer feature-representation ablations on collected v3 traces.")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "collected_v3")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(int(args.seed))
    root = args.root.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else root / "collected_v3_feature_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_full = load_npz(root / "collected_train_v3" / "collected_train_v3.npz")
    val_full = load_npz(root / "collected_val_v3" / "collected_val_v3.npz")
    test_full = load_npz(root / "collected_test_v3" / "collected_test_v3.npz")
    feature_names = [str(x) for x in train_full["feature_names"].tolist()]

    n_train = len(train_full["y_risk"])
    n_val = len(val_full["y_risk"])
    split = SplitIndex(
        train=np.arange(0, n_train, dtype=np.int64),
        val=np.arange(n_train, n_train + n_val, dtype=np.int64),
        test=np.asarray([], dtype=np.int64),
    )

    labels = {
        "y_risk": np.concatenate([train_full["y_risk"], val_full["y_risk"]], axis=0).astype(np.int64),
        "time_to_risk": np.concatenate([train_full["time_to_risk"], val_full["time_to_risk"]], axis=0).astype(np.int64),
        "risk_intensity": np.concatenate([train_full["risk_intensity"], val_full["risk_intensity"]], axis=0).astype(np.float32),
        "state_cls": np.concatenate([train_full["state_cls"], val_full["state_cls"]], axis=0).astype(np.int64),
        "current_normal": np.concatenate([train_full["current_normal"], val_full["current_normal"]], axis=0).astype(np.int64),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "train_npz": str(root / "collected_train_v3" / "collected_train_v3.npz"),
        "val_npz": str(root / "collected_val_v3" / "collected_val_v3.npz"),
        "test_npz": str(root / "collected_test_v3" / "collected_test_v3.npz"),
        "experiments": [
            {
                **exp,
                "feature_names": feature_names if exp["feature_indices"] is None else [feature_names[i] for i in exp["feature_indices"]],
            }
            for exp in EXPERIMENTS
        ],
        "device": str(device),
        "args": vars(args) | {"root": str(root), "out_dir": str(out_dir)},
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    all_metrics: list[dict] = []
    train_meta: dict[str, dict] = {}

    for exp in EXPERIMENTS:
        model_name = str(exp["model"])
        indices = exp["feature_indices"]
        print(f"\n===== Training {model_name} =====")
        train = select_features(train_full, indices)
        val = select_features(val_full, indices)
        test = select_features(test_full, indices)
        combined = make_combined(train, val)

        model, scaler, threshold, meta = train_one_model(
            model_name,
            combined["X_opr"].astype(np.float32),
            labels,
            split,
            horizon=int(train_full["horizon"]),
            args=args,
            out_dir=out_dir,
            device=device,
        )
        selected_names = feature_names if indices is None else [feature_names[i] for i in indices]
        meta["feature_names"] = selected_names
        train_meta[model_name] = meta
        rows = eval_one_model(model_name, model, scaler, threshold, test, args, out_dir, device)
        for row in rows:
            row["feature_set"] = model_name.replace("TSMixer-", "")
            row["input_dim"] = len(selected_names)
        all_metrics.extend(rows)

    metrics = pd.DataFrame(all_metrics)
    metrics = metrics[["model", "feature_set", "input_dim", "subset"] + [c for c in metrics.columns if c not in {"model", "feature_set", "input_dim", "subset"}]]
    metrics.to_csv(out_dir / "metrics_summary.csv", index=False)
    (out_dir / "metrics_summary.json").write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "train_meta_all.json").write_text(json.dumps(train_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    key = metrics[metrics["subset"].isin(["overall_test", "current_normal_test", "predictable_onset_vs_normal", "recovery_false_positive"])]
    cols = ["model", "feature_set", "input_dim", "subset", "n", "positives", "pr_auc", "false_positive_rate", "p_at_100", "recall_at_100"]
    print("\n===== TSMixer Feature Ablation Metrics =====")
    print(key[[c for c in cols if c in key.columns]].to_string(index=False))
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
