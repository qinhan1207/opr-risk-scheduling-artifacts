from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_rote_models_round2 import SplitIndex, load_npz, set_seed, train_one_model
from train_collected_v3 import eval_one_model


GROUPS = {
    "local_deviation": ["residual", "abs_residual"],
    "threshold_proximity": ["distance_to_threshold", "near_threshold_ratio"],
    "positive_growth": ["pos_delta_latency", "pos_delta_std", "pos_delta_sum"],
    "near_threshold_persistence": ["near_threshold_count_ratio"],
    "local_peak": ["rolling_max_ratio"],
}

DISPLAY_NAMES = {
    "full_opr": "Full OPR",
    "wo_local_deviation": "w/o local deviation",
    "wo_threshold_proximity": "w/o threshold proximity",
    "wo_positive_growth": "w/o positive growth",
    "wo_near_threshold_persistence": "w/o near-threshold persistence",
    "wo_local_peak": "w/o local peak",
}

KEY_SUBSETS = [
    "overall_test",
    "current_normal_test",
    "predictable_onset_vs_normal",
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


def feature_indices_for_variant(feature_names: list[str], remove_group: str | None) -> list[int]:
    if remove_group is None:
        return list(range(len(feature_names)))
    remove = set(GROUPS[remove_group])
    return [i for i, name in enumerate(feature_names) if name not in remove]


def select_features(data: dict, indices: list[int]) -> dict:
    selected = dict(data)
    selected["X_opr"] = data["X_opr"].astype(np.float32)[:, :, indices]
    return selected


def build_variants(feature_names: list[str]) -> list[dict]:
    variants = [
        {
            "variant": "full_opr",
            "model": "TSMixer-OPRGroup-full",
            "remove_group": None,
            "feature_indices": feature_indices_for_variant(feature_names, None),
        }
    ]
    for group in GROUPS:
        variant = f"wo_{group}"
        variants.append(
            {
                "variant": variant,
                "model": f"TSMixer-OPRGroup-{variant}",
                "remove_group": group,
                "feature_indices": feature_indices_for_variant(feature_names, group),
            }
        )
    return variants


def compact_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in metrics.groupby("variant", sort=False):
        by_subset = {row["subset"]: row for _, row in group.iterrows()}
        overall = by_subset.get("overall_test")
        cn = by_subset.get("current_normal_test")
        pred = by_subset.get("predictable_onset_vs_normal")
        rec = by_subset.get("recovery_false_positive")
        rows.append(
            {
                "input_variant": DISPLAY_NAMES.get(variant, variant),
                "overall_pr_auc": float(overall["pr_auc"]) if overall is not None else float("nan"),
                "current_normal_pr_auc": float(cn["pr_auc"]) if cn is not None else float("nan"),
                "predictable_onset_pr_auc": float(pred["pr_auc"]) if pred is not None else float("nan"),
                "brier": float(overall["brier"]) if overall is not None else float("nan"),
                "recovery_fpr": float(rec["false_positive_rate"]) if rec is not None else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run OPR feature-group ablation with fixed TSMixer on collected v3 traces.")
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
    out_dir = args.out_dir.resolve() if args.out_dir is not None else root / "collected_v3_opr_group_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_full = load_npz(root / "collected_train_v3" / "collected_train_v3.npz")
    val_full = load_npz(root / "collected_val_v3" / "collected_val_v3.npz")
    test_full = load_npz(root / "collected_test_v3" / "collected_test_v3.npz")
    feature_names = [str(x) for x in train_full["feature_names"].tolist()]
    variants = build_variants(feature_names)

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
        "feature_names": feature_names,
        "groups": GROUPS,
        "variants": [
            {
                **variant,
                "display_name": DISPLAY_NAMES[variant["variant"]],
                "selected_features": [feature_names[i] for i in variant["feature_indices"]],
            }
            for variant in variants
        ],
        "device": str(device),
        "args": vars(args) | {"root": str(root), "out_dir": str(out_dir)},
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    all_metrics: list[dict] = []
    train_meta: dict[str, dict] = {}

    for variant in variants:
        model_name = str(variant["model"])
        indices = list(variant["feature_indices"])
        print(f"\n===== Training {DISPLAY_NAMES[variant['variant']]} =====")
        set_seed(int(args.seed))
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
        selected_features = [feature_names[i] for i in indices]
        meta["variant"] = variant["variant"]
        meta["display_name"] = DISPLAY_NAMES[variant["variant"]]
        meta["remove_group"] = variant["remove_group"]
        meta["feature_names"] = selected_features
        train_meta[variant["variant"]] = meta

        rows = eval_one_model(model_name, model, scaler, threshold, test, args, out_dir, device)
        for row in rows:
            row["variant"] = variant["variant"]
            row["input_variant"] = DISPLAY_NAMES[variant["variant"]]
            row["remove_group"] = variant["remove_group"] or ""
            row["input_dim"] = len(selected_features)
        all_metrics.extend(rows)

    metrics = pd.DataFrame(all_metrics)
    metrics = metrics[
        ["variant", "input_variant", "remove_group", "input_dim", "model", "subset"]
        + [c for c in metrics.columns if c not in {"variant", "input_variant", "remove_group", "input_dim", "model", "subset"}]
    ]
    metrics.to_csv(out_dir / "metrics_summary.csv", index=False)
    (out_dir / "metrics_summary.json").write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "train_meta_all.json").write_text(json.dumps(train_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    table = compact_table(metrics[metrics["subset"].isin(KEY_SUBSETS)])
    table.to_csv(out_dir / "table_opr_group_ablation.csv", index=False)

    print("\n===== Table X. Ablation study of OPR feature groups =====")
    print(table.to_string(index=False))
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
