from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "raw_latency",
    "delta_latency",
    "rolling_mean",
    "rolling_std",
    "residual",
    "abs_residual",
    "distance_to_threshold",
    "near_threshold_ratio",
    "pos_delta_latency",
    "pos_delta_std",
    "rolling_max_ratio",
    "pos_delta_sum",
    "near_threshold_count_ratio",
]

STATE_TO_ID = {
    "stable": 0,
    "onset": 1,
    "persistent": 2,
    "recovery": 3,
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    return pd.read_csv(path)


def _latency_column(df: pd.DataFrame) -> str:
    if "Raw_Latency" in df.columns:
        return "Raw_Latency"
    if "latency_ms" in df.columns:
        return "latency_ms"
    raise ValueError("CSV must contain Raw_Latency or latency_ms")


def _sort_trace(df: pd.DataFrame) -> pd.DataFrame:
    if "Timestamp" in df.columns:
        return df.sort_values("Timestamp").reset_index(drop=True)
    if "t" in df.columns:
        return df.sort_values("t").reset_index(drop=True)
    return df.reset_index(drop=True)


def build_opr_features(latency: np.ndarray, tau_net: float, rolling_window: int) -> np.ndarray:
    latency = np.asarray(latency, dtype=np.float64)
    s = pd.Series(latency)

    delta = s.diff().fillna(0.0).to_numpy(dtype=np.float64)
    rolling_mean = (
        s.rolling(window=int(rolling_window), min_periods=1)
        .mean()
        .to_numpy(dtype=np.float64)
    )
    rolling_std = (
        s.rolling(window=int(rolling_window), min_periods=2)
        .std(ddof=0)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    residual = latency - rolling_mean
    std_delta = pd.Series(rolling_std).diff().fillna(0.0).to_numpy(dtype=np.float64)
    pos_delta = np.maximum(delta, 0.0)
    rolling_max = (
        s.rolling(window=int(rolling_window), min_periods=1)
        .max()
        .to_numpy(dtype=np.float64)
    )
    pos_delta_sum = (
        pd.Series(pos_delta)
        .rolling(window=int(rolling_window), min_periods=1)
        .sum()
        .to_numpy(dtype=np.float64)
    )
    near_threshold_count_ratio = (
        (s >= 0.8 * float(tau_net))
        .astype(float)
        .rolling(window=int(rolling_window), min_periods=1)
        .mean()
        .to_numpy(dtype=np.float64)
    )

    features = np.column_stack(
        [
            latency,
            delta,
            rolling_mean,
            rolling_std,
            residual,
            np.abs(residual),
            float(tau_net) - latency,
            latency / max(float(tau_net), 1e-6),
            pos_delta,
            np.maximum(std_delta, 0.0),
            rolling_max / max(float(tau_net), 1e-6),
            pos_delta_sum,
            near_threshold_count_ratio,
        ]
    )
    return features.astype(np.float32)


def build_opr_features_for_trace(
    df: pd.DataFrame,
    latency: np.ndarray,
    tau_net: float,
    rolling_window: int,
) -> np.ndarray:
    if "scenario_id" not in df.columns:
        return build_opr_features(
            latency,
            tau_net=tau_net,
            rolling_window=rolling_window,
        )

    features = np.zeros((len(df), len(FEATURE_NAMES)), dtype=np.float32)
    for _, idx in df.groupby("scenario_id", sort=False).groups.items():
        pos = np.asarray(list(idx), dtype=int)
        features[pos] = build_opr_features(
            latency[pos],
            tau_net=tau_net,
            rolling_window=rolling_window,
        )
    return features


def _state_name(current_risk: bool, future_risk: bool) -> str:
    if (not current_risk) and (not future_risk):
        return "stable"
    if (not current_risk) and future_risk:
        return "onset"
    if current_risk and future_risk:
        return "persistent"
    return "recovery"


def _meta_value(row: pd.Series, name: str, default: object = "") -> object:
    if name in row.index:
        value = row[name]
        if pd.isna(value):
            return default
        return value
    return default


def _future_onset_type(future_rows: pd.DataFrame, first_risk_offset: int | None) -> str:
    if first_risk_offset is None:
        return "none"

    row = future_rows.iloc[int(first_risk_offset)]
    block_type = _meta_value(row, "block_type", "")
    if block_type == "predictable_onset_block":
        return "predictable-onset"
    if block_type == "abrupt_onset_block":
        return "abrupt-onset"
    if block_type == "persistent_risk_block":
        return "persistent-risk-entry"
    if block_type == "recovery_block":
        return "recovery-risk-entry"
    if block_type == "gradual_onset_block":
        return "gradual"
    if block_type == "volatility_onset_block":
        return "volatility"
    if block_type == "microspike_onset_block":
        return "microspike"
    if block_type:
        return str(block_type)
    return "unknown"


def build_rote_dataset(
    df: pd.DataFrame,
    source_name: str,
    lookback: int,
    horizon: int,
    tau_net: float,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    df = _sort_trace(df)
    lat_col = _latency_column(df)
    latency = df[lat_col].astype(float).to_numpy(dtype=np.float64)
    features = build_opr_features_for_trace(
        df,
        latency=latency,
        tau_net=tau_net,
        rolling_window=lookback,
    )

    n = len(df)
    if n < lookback + horizon:
        raise ValueError(
            f"Trace too short for lookback={lookback}, horizon={horizon}: rows={n}"
        )

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    time_to_risk_list: list[int] = []
    risk_intensity_list: list[float] = []
    state_cls_list: list[int] = []
    current_normal_list: list[int] = []
    meta_rows: list[dict[str, object]] = []
    skipped_cross_scenario = 0

    for end in range(int(lookback) - 1, n - int(horizon)):
        start = end - int(lookback) + 1
        window_df = df.iloc[start : end + 1]
        future_rows = df.iloc[end + 1 : end + 1 + int(horizon)]

        if "scenario_id" in df.columns:
            current_scenario = df.iloc[end]["scenario_id"]
            if not window_df["scenario_id"].eq(current_scenario).all():
                skipped_cross_scenario += 1
                continue
            if not future_rows["scenario_id"].eq(current_scenario).all():
                skipped_cross_scenario += 1
                continue

        current_latency = float(latency[end])
        future = future_rows[lat_col].astype(float).to_numpy(dtype=np.float64)
        future_risk_mask = future > float(tau_net)

        current_risk = current_latency > float(tau_net)
        future_risk = bool(np.any(future_risk_mask))
        state_name = _state_name(current_risk=current_risk, future_risk=future_risk)

        if future_risk:
            first_risk_offset = int(np.argmax(future_risk_mask))
            time_to_risk = int(first_risk_offset + 1)
            future_first_risk_idx = int(end + 1 + first_risk_offset)
            future_first_risk_latency = float(future[first_risk_offset])
        else:
            first_risk_offset = None
            time_to_risk = int(horizon) + 1
            future_first_risk_idx = -1
            future_first_risk_latency = float("nan")

        max_future = float(np.max(future))
        risk_intensity = max(0.0, (max_future - float(tau_net)) / float(tau_net))
        state_cls = STATE_TO_ID[state_name]
        current_normal = int(not current_risk)
        y_risk = int(future_risk)
        future_onset_type = _future_onset_type(future_rows, first_risk_offset)

        X_list.append(features[start : end + 1])
        y_list.append(y_risk)
        time_to_risk_list.append(time_to_risk)
        risk_intensity_list.append(risk_intensity)
        state_cls_list.append(state_cls)
        current_normal_list.append(current_normal)

        row = df.iloc[end]
        onset_type = _meta_value(row, "onset_type", "")
        meta_rows.append(
            {
                "source": source_name,
                "end_idx": int(end),
                "start_idx": int(start),
                "sample_type": state_name,
                "state_name": state_name,
                "y_risk": y_risk,
                "time_to_risk": time_to_risk,
                "risk_intensity": risk_intensity,
                "current_normal": current_normal,
                "state_cls": state_cls,
                "future_onset_type": future_onset_type,
                "onset_type": onset_type,
                "future_first_risk_idx": future_first_risk_idx,
                "future_first_risk_latency": future_first_risk_latency,
                "end_time": _meta_value(row, "Timestamp", _meta_value(row, "t", end)),
                "current_latency": current_latency,
                "future_max_latency": max_future,
                "scenario_id": _meta_value(row, "scenario_id", -1),
                "block_type": _meta_value(row, "block_type", ""),
                "phase": _meta_value(row, "phase", ""),
            }
        )

    X_opr = np.stack(X_list, axis=0).astype(np.float32)
    y_risk = np.asarray(y_list, dtype=np.int64)
    time_to_risk = np.asarray(time_to_risk_list, dtype=np.int64)
    risk_intensity = np.asarray(risk_intensity_list, dtype=np.float32)
    state_cls = np.asarray(state_cls_list, dtype=np.int64)
    current_normal = np.asarray(current_normal_list, dtype=np.int64)
    meta = pd.DataFrame(meta_rows)

    payload = {
        "X_opr": X_opr,
        "y_risk": y_risk,
        "time_to_risk": time_to_risk,
        "risk_intensity": risk_intensity,
        "state_cls": state_cls,
        "current_normal": current_normal,
        "meta": meta.to_records(index=False),
        "feature_names": np.asarray(FEATURE_NAMES, dtype=object),
        "state_names": np.asarray(["stable", "onset", "persistent", "recovery"], dtype=object),
        "lookback": np.asarray(int(lookback), dtype=np.int64),
        "horizon": np.asarray(int(horizon), dtype=np.int64),
        "tau_net": np.asarray(float(tau_net), dtype=np.float32),
        "skipped_cross_scenario": np.asarray(int(skipped_cross_scenario), dtype=np.int64),
    }
    return payload, meta


def save_dataset(payload: dict[str, np.ndarray], meta: pd.DataFrame, out_npz: Path) -> dict:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **payload)

    meta_csv = out_npz.with_suffix(".meta.csv")
    meta.to_csv(meta_csv, index=False, encoding="utf-8-sig")

    state_counts = {
        name: int(np.sum(payload["state_cls"] == idx))
        for name, idx in STATE_TO_ID.items()
    }
    summary = {
        "npz": str(out_npz),
        "meta_csv": str(meta_csv),
        "n_samples": int(payload["X_opr"].shape[0]),
        "lookback": int(payload["lookback"]),
        "horizon": int(payload["horizon"]),
        "tau_net": float(payload["tau_net"]),
        "feature_names": list(FEATURE_NAMES),
        "state_to_id": dict(STATE_TO_ID),
        "positive_y_risk": int(np.sum(payload["y_risk"] == 1)),
        "state_counts": state_counts,
        "current_normal": int(np.sum(payload["current_normal"] == 1)),
        "skipped_cross_scenario": int(payload["skipped_cross_scenario"]),
    }

    summary_path = out_npz.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build ROTE-Net OPR datasets from collected latency traces.")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--lookback", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--tau-net", type=float, default=45.0)
    args = ap.parse_args()

    root = args.root.resolve()
    out_root = args.out_root.resolve() if args.out_root is not None else Path(__file__).resolve().parent / "collected_v3"
    jobs = [
        (
            "collected_train_v3",
            root / "data" / "raw_trace" / "train_v3_raw_trace.csv",
            out_root / "collected_train_v3" / "collected_train_v3.npz",
        ),
        (
            "collected_test_v3",
            root / "data" / "raw_trace" / "test_v3_raw_trace.csv",
            out_root / "collected_test_v3" / "collected_test_v3.npz",
        ),
    ]

    summaries = []
    for source_name, in_csv, out_npz in jobs:
        df = _read_csv(in_csv)
        payload, meta = build_rote_dataset(
            df,
            source_name=source_name,
            lookback=int(args.lookback),
            horizon=int(args.horizon),
            tau_net=float(args.tau_net),
        )
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        summary = save_dataset(payload, meta, out_npz)
        summary["input_csv"] = str(in_csv)
        summaries.append(summary)

    out_root.mkdir(parents=True, exist_ok=True)
    combined_path = out_root / "collected_v3_build_summary.json"
    combined_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    for summary in summaries:
        print(f"Wrote {summary['npz']}")
        print(
            f"  samples={summary['n_samples']} positives={summary['positive_y_risk']} "
            f"states={summary['state_counts']}"
        )
    print(f"Wrote {combined_path}")


if __name__ == "__main__":
    main()
