from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_rote_models_round2 import (
    MODEL_NAMES as _ROUND2_NAMES,
    SplitIndex,
    eval_model_on_array,
    load_npz,
    metrics_for_subset,
    precision_at_k,
    recarray_to_frame,
    set_seed,
    train_one_model,
)


MODEL_NAMES = [
    "OPR-GRU-v2",
    "OPR-ModernTCN-v2",
    "OPR-TSMixer-v2",
    "OPR-xLSTM-v2",
]
PREDICTABLE_TYPES = {"gradual", "volatility", "microspike"}


def recall_at_k(y_true: np.ndarray, prob: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true, dtype=int)
    if len(y_true) == 0 or int(y_true.sum()) == 0:
        return float("nan")
    k = min(int(k), len(y_true))
    idx = np.argsort(-np.asarray(prob, dtype=float))[:k]
    return float(y_true[idx].sum() / y_true.sum())


def add_rank_metrics(row: dict, y_true: np.ndarray, prob: np.ndarray) -> dict:
    for k in [50, 100, 200]:
        row[f"p_at_{k}"] = precision_at_k(y_true, prob, k)
        row[f"recall_at_{k}"] = recall_at_k(y_true, prob, k)
    return row


def subset_metrics(y: np.ndarray, prob: np.ndarray, threshold: float, name: str) -> dict:
    row = metrics_for_subset(y, prob, threshold, name)
    return add_rank_metrics(row, y, prob)


def eval_one_model(
    model_name: str,
    model: torch.nn.Module,
    scaler,
    threshold: float,
    test: dict,
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
) -> list[dict]:
    meta = recarray_to_frame(test["meta"])
    y = test["y_risk"].astype(int)
    prob = eval_model_on_array(
        model,
        scaler,
        threshold,
        model_name,
        test["X_opr"].astype(np.float32),
        y,
        int(test["horizon"]),
        device,
        args,
        test["time_to_risk"],
        test["risk_intensity"],
        test["state_cls"],
        test["current_normal"],
    )

    current_normal = meta["current_normal"].astype(int).to_numpy() == 1
    state_name = meta["state_name"].astype(str).to_numpy()
    onset_type = meta["onset_type"].astype(str).to_numpy()
    cn_negative = current_normal & (y == 0)
    cn_onset = current_normal & (state_name == "onset")
    predictable_pos = cn_onset & np.isin(onset_type, list(PREDICTABLE_TYPES))
    abrupt_pos = cn_onset & (onset_type == "abrupt")
    recovery_mask = state_name == "recovery"

    rows = [
        subset_metrics(y, prob, threshold, "overall_test"),
        subset_metrics(y[current_normal], prob[current_normal], threshold, "current_normal_test"),
        subset_metrics(predictable_pos[cn_negative | predictable_pos].astype(int), prob[cn_negative | predictable_pos], threshold, "predictable_onset_vs_normal"),
        subset_metrics(abrupt_pos[cn_negative | abrupt_pos].astype(int), prob[cn_negative | abrupt_pos], threshold, "abrupt_onset_vs_normal"),
    ]

    if np.any(recovery_mask):
        rec_prob = prob[recovery_mask]
        rows.append(
            {
                "subset": "recovery_false_positive",
                "n": int(np.sum(recovery_mask)),
                "positives": 0,
                "positive_rate": 0.0,
                "roc_auc": float("nan"),
                "pr_auc": float("nan"),
                "brier": float(np.mean(rec_prob**2)),
                "threshold": float(threshold),
                "f1": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "false_positive_rate": float(np.mean(rec_prob >= float(threshold))),
                "mean_prob": float(np.mean(rec_prob)),
                "p_at_50": float("nan"),
                "recall_at_50": float("nan"),
                "p_at_100": float("nan"),
                "recall_at_100": float("nan"),
                "p_at_200": float("nan"),
                "recall_at_200": float("nan"),
            }
        )

    for row in rows:
        row["model"] = model_name

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "prob": prob,
            "y_risk": y,
            "current_normal": meta["current_normal"].astype(int).to_numpy(),
            "state_name": state_name,
            "onset_type": onset_type,
            "block_type": meta["block_type"].astype(str).to_numpy(),
            "phase": meta["phase"].astype(str).to_numpy(),
        }
    ).to_csv(pred_dir / f"{model_name}_collected_v3_test_predictions.csv", index=False)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and evaluate collected-trace v3 mechanism-validation models.")
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
    out_dir = args.out_dir.resolve() if args.out_dir is not None else root / "collected_v3_training_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_npz(root / "collected_train_v3" / "collected_train_v3.npz")
    val = load_npz(root / "collected_val_v3" / "collected_val_v3.npz")
    test = load_npz(root / "collected_test_v3" / "collected_test_v3.npz")

    combined = {
        "X_opr": np.concatenate([train["X_opr"], val["X_opr"]], axis=0),
        "y_risk": np.concatenate([train["y_risk"], val["y_risk"]], axis=0),
        "time_to_risk": np.concatenate([train["time_to_risk"], val["time_to_risk"]], axis=0),
        "risk_intensity": np.concatenate([train["risk_intensity"], val["risk_intensity"]], axis=0),
        "state_cls": np.concatenate([train["state_cls"], val["state_cls"]], axis=0),
        "current_normal": np.concatenate([train["current_normal"], val["current_normal"]], axis=0),
    }
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
        "models": MODEL_NAMES,
        "device": str(device),
        "args": vars(args) | {"root": str(root), "out_dir": str(out_dir)},
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    all_metrics: list[dict] = []
    train_meta: dict[str, dict] = {}
    for model_name in MODEL_NAMES:
        print(f"\n===== Training {model_name} =====")
        model, scaler, threshold, meta = train_one_model(
            model_name,
            combined["X_opr"].astype(np.float32),
            labels,
            split,
            horizon=int(train["horizon"]),
            args=args,
            out_dir=out_dir,
            device=device,
        )
        train_meta[model_name] = meta
        all_metrics.extend(eval_one_model(model_name, model, scaler, threshold, test, args, out_dir, device))

    metrics = pd.DataFrame(all_metrics)
    metrics = metrics[["model", "subset"] + [c for c in metrics.columns if c not in {"model", "subset"}]]
    metrics.to_csv(out_dir / "metrics_summary.csv", index=False)
    (out_dir / "metrics_summary.json").write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "train_meta_all.json").write_text(json.dumps(train_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== Collected v3 Metrics =====")
    cols = ["model", "subset", "n", "positives", "roc_auc", "pr_auc", "brier", "f1", "precision", "recall", "p_at_50", "recall_at_50", "p_at_100", "recall_at_100", "false_positive_rate"]
    available = [c for c in cols if c in metrics.columns]
    print(metrics[available].to_string(index=False))
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
