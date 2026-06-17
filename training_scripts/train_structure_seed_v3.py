from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_rote_models_round2 import SplitIndex, load_npz, set_seed, train_one_model
from train_collected_v3 import eval_one_model


DEFAULT_MODELS = ["OPR-GRU-v2", "OPR-TSMixer-v2"]
DEFAULT_SEEDS = [42, 43, 44]
KEY_SUBSETS = [
    "overall_test",
    "current_normal_test",
    "predictable_onset_vs_normal",
    "abrupt_onset_vs_normal",
    "recovery_false_positive",
]


def make_combined(train: dict, val: dict) -> dict:
    return {
        "X_opr": np.concatenate([train["X_opr"], val["X_opr"]], axis=0),
        "y_risk": np.concatenate([train["y_risk"], val["y_risk"]], axis=0),
        "time_to_risk": np.concatenate([train["time_to_risk"], val["time_to_risk"]], axis=0),
        "risk_intensity": np.concatenate([train["risk_intensity"], val["risk_intensity"]], axis=0),
        "state_cls": np.concatenate([train["state_cls"], val["state_cls"]], axis=0),
        "current_normal": np.concatenate([train["current_normal"], val["current_normal"]], axis=0),
    }


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = metrics[metrics["subset"].isin(KEY_SUBSETS)].copy()
    value_cols = ["pr_auc", "false_positive_rate", "p_at_100", "recall_at_100"]
    available = [c for c in value_cols if c in rows.columns]
    grouped = rows.groupby(["base_model", "subset"], as_index=False)[available].agg(["mean", "std"])
    grouped.columns = [
        "_".join([str(x) for x in col if str(x)])
        if isinstance(col, tuple)
        else str(col)
        for col in grouped.columns
    ]
    return grouped


def main() -> None:
    ap = argparse.ArgumentParser(description="Run multi-seed structure comparison for selected collected-trace v3 models.")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "collected_v3")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    args = ap.parse_args()

    root = args.root.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else root / "collected_v3_structure_seeds"
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_npz(root / "collected_train_v3" / "collected_train_v3.npz")
    val = load_npz(root / "collected_val_v3" / "collected_val_v3.npz")
    test = load_npz(root / "collected_test_v3" / "collected_test_v3.npz")
    combined = make_combined(train, val)

    n_train = len(train["y_risk"])
    n_val = len(val["y_risk"])
    split = SplitIndex(
        train=np.arange(0, n_train, dtype=np.int64),
        val=np.arange(n_train, n_train + n_val, dtype=np.int64),
        test=np.asarray([], dtype=np.int64),
    )
    labels = {
        "y_risk": combined["y_risk"].astype(np.int64),
        "time_to_risk": combined["time_to_risk"].astype(np.int64),
        "risk_intensity": combined["risk_intensity"].astype(np.float32),
        "state_cls": combined["state_cls"].astype(np.int64),
        "current_normal": combined["current_normal"].astype(np.int64),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "train_npz": str(root / "collected_train_v3" / "collected_train_v3.npz"),
        "val_npz": str(root / "collected_val_v3" / "collected_val_v3.npz"),
        "test_npz": str(root / "collected_test_v3" / "collected_test_v3.npz"),
        "models": list(args.models),
        "seeds": [int(s) for s in args.seeds],
        "device": str(device),
        "args": vars(args) | {"root": str(root), "out_dir": str(out_dir)},
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    all_metrics: list[dict] = []
    train_meta: dict[str, dict] = {}

    for seed in args.seeds:
        for model_name in args.models:
            run_name = f"{model_name}-seed{int(seed)}"
            print(f"\n===== Training {run_name} =====")
            set_seed(int(seed))
            model, scaler, threshold, meta = train_one_model(
                model_name,
                combined["X_opr"].astype(np.float32),
                labels,
                split,
                horizon=int(train["horizon"]),
                args=args,
                out_dir=out_dir / f"seed_{int(seed)}",
                device=device,
            )
            train_meta[run_name] = meta
            rows = eval_one_model(run_name, model, scaler, threshold, test, args, out_dir, device)
            for row in rows:
                row["base_model"] = model_name
                row["seed"] = int(seed)
            all_metrics.extend(rows)

    metrics = pd.DataFrame(all_metrics)
    metrics = metrics[["base_model", "seed", "model", "subset"] + [c for c in metrics.columns if c not in {"base_model", "seed", "model", "subset"}]]
    metrics.to_csv(out_dir / "metrics_by_seed.csv", index=False)
    (out_dir / "metrics_by_seed.json").write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "train_meta_all.json").write_text(json.dumps(train_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = aggregate_metrics(metrics)
    summary.to_csv(out_dir / "metrics_seed_summary.csv", index=False)

    print("\n===== Multi-seed Summary =====")
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
