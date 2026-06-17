from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch
from flask import Flask, jsonify, request
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("OPR_MODEL_DIR", REPO_ROOT / "models" / "opr-risk"))
MODEL_PATH = MODEL_DIR / "model.pt"
SCALER_PATH = MODEL_DIR / "scaler.json"
META_PATH = MODEL_DIR / "train_meta.json"

PORT = int(os.getenv("OPR_INFERENCE_PORT", "5001"))
TAU_NET = float(os.getenv("OPR_TAU_NET_MS", "45.0"))
ROLLING_WINDOW = int(os.getenv("OPR_ROLLING_WINDOW", "10"))

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

app = Flask(__name__)

_buffers_lock = Lock()
_cluster_buffers: dict[str, deque[float]] = {}
_cluster_last_seen: dict[str, float] = {}


class TSMixerBlock(nn.Module):
    def __init__(
        self,
        lookback: int,
        feature_dim: int,
        token_hidden: int = 32,
        feature_hidden: int = 64,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.token_norm = nn.Identity() if feature_dim == 1 else nn.LayerNorm(feature_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(lookback, token_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_hidden, lookback),
            nn.Dropout(dropout),
        )
        self.feature_norm = nn.Identity() if feature_dim == 1 else nn.LayerNorm(feature_dim)
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_hidden, feature_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.token_norm(x).transpose(1, 2)
        x = x + self.token_mlp(y).transpose(1, 2)
        return x + self.feature_mlp(self.feature_norm(x))


class OprTSMixer(nn.Module):
    def __init__(self, input_dim: int = 13, lookback: int = 10, depth: int = 3, dropout: float = 0.15) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                TSMixerBlock(
                    lookback=lookback,
                    feature_dim=input_dim,
                    token_hidden=max(16, lookback * 2),
                    feature_hidden=64,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.head = nn.Sequential(
            nn.Identity() if input_dim == 1 else nn.LayerNorm(input_dim),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(lookback * input_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.blocks(x)
        return {"risk_logit": self.head(z).squeeze(-1)}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


meta = _load_json(META_PATH)
scaler = _load_json(SCALER_PATH)

MODEL_NAME = str(meta.get("model_name", "OPR-TSMixer-v2"))
INPUT_DIM = int(meta.get("input_dim", len(FEATURE_NAMES)))
LOOKBACK = int(meta.get("lookback", 10))
THRESHOLD = float(meta.get("val_threshold_best_f1", 0.5))

if INPUT_DIM != len(FEATURE_NAMES):
    raise ValueError(f"Model input_dim={INPUT_DIM}, but server defines {len(FEATURE_NAMES)} features")

scaler_mean = np.asarray(scaler["mean"], dtype=np.float32)
scaler_std = np.asarray(scaler["std"], dtype=np.float32)
scaler_std = np.where(np.abs(scaler_std) < 1e-6, 1.0, scaler_std)

device = torch.device("cpu")
model = OprTSMixer(input_dim=INPUT_DIM, lookback=LOOKBACK).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


def _get_cluster_name(req_json: dict[str, Any]) -> str | None:
    header_value = request.headers.get("X-Cluster-Name")
    if header_value and str(header_value).strip():
        return str(header_value).strip()

    for key in ("cluster", "cluster_name", "clusterName"):
        value = req_json.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _get_buffer(cluster_name: str) -> deque[float]:
    with _buffers_lock:
        buffer = _cluster_buffers.get(cluster_name)
        if buffer is None or getattr(buffer, "maxlen", None) != LOOKBACK:
            buffer = deque(maxlen=LOOKBACK)
            _cluster_buffers[cluster_name] = buffer
        _cluster_last_seen[cluster_name] = time.time()
        return buffer


def _as_latency_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        raise ValueError("'latency_window' must be a list of numbers")

    latencies: list[float] = []
    for idx, item in enumerate(value):
        try:
            latency = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"latency_window[{idx}] is not numeric") from exc
        if not np.isfinite(latency):
            raise ValueError(f"latency_window[{idx}] is not finite")
        latencies.append(latency)
    return latencies


def build_opr_features(latencies: list[float]) -> np.ndarray:
    latency = np.asarray(latencies, dtype=np.float64)
    n = latency.shape[0]
    features = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)

    delta = np.zeros(n, dtype=np.float64)
    if n > 1:
        delta[1:] = latency[1:] - latency[:-1]

    rolling_mean = np.zeros(n, dtype=np.float64)
    rolling_std = np.zeros(n, dtype=np.float64)
    rolling_max = np.zeros(n, dtype=np.float64)
    pos_delta_sum = np.zeros(n, dtype=np.float64)
    near_threshold_count_ratio = np.zeros(n, dtype=np.float64)

    pos_delta = np.maximum(delta, 0.0)
    near_threshold = (latency >= 0.8 * TAU_NET).astype(np.float64)

    for i in range(n):
        start = max(0, i - ROLLING_WINDOW + 1)
        lat_slice = latency[start : i + 1]
        rolling_mean[i] = float(np.mean(lat_slice))
        rolling_std[i] = float(np.std(lat_slice, ddof=0)) if len(lat_slice) >= 2 else 0.0
        rolling_max[i] = float(np.max(lat_slice))
        pos_delta_sum[i] = float(np.sum(pos_delta[start : i + 1]))
        near_threshold_count_ratio[i] = float(np.mean(near_threshold[start : i + 1]))

    residual = latency - rolling_mean
    std_delta = np.zeros(n, dtype=np.float64)
    if n > 1:
        std_delta[1:] = rolling_std[1:] - rolling_std[:-1]

    tau = max(TAU_NET, 1e-6)
    features[:, 0] = latency
    features[:, 1] = delta
    features[:, 2] = rolling_mean
    features[:, 3] = rolling_std
    features[:, 4] = residual
    features[:, 5] = np.abs(residual)
    features[:, 6] = TAU_NET - latency
    features[:, 7] = latency / tau
    features[:, 8] = pos_delta
    features[:, 9] = np.maximum(std_delta, 0.0)
    features[:, 10] = rolling_max / tau
    features[:, 11] = pos_delta_sum
    features[:, 12] = near_threshold_count_ratio
    return features.astype(np.float32)


def predict_from_latency_window(latencies: list[float]) -> tuple[float, bool]:
    if len(latencies) != LOOKBACK:
        raise ValueError(f"latency_window size {len(latencies)} does not match lookback {LOOKBACK}")

    raw_features = build_opr_features(latencies)
    scaled = (raw_features - scaler_mean) / scaler_std
    x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)["risk_logit"]
        prob = torch.sigmoid(logits).item()

    return float(prob), bool(prob >= THRESHOLD)


@app.route("/predict/window", methods=["POST"])
def predict_window():
    try:
        req = request.get_json(silent=True)
        if not isinstance(req, dict):
            return jsonify({"code": 400, "error": "Invalid JSON body"}), 400

        cluster_name = _get_cluster_name(req)
        if not cluster_name:
            return jsonify({"code": 400, "error": "Missing cluster name"}), 400

        latencies = _as_latency_list(req.get("latency_window"))
        if len(latencies) < LOOKBACK:
            return jsonify({
                "code": 202,
                "msg": f"Buffering... ({len(latencies)}/{LOOKBACK})",
                "prob": 0.0,
                "is_fault": False,
                "cluster": cluster_name,
                "mode": "latency_window",
            }), 202

        latencies = latencies[-LOOKBACK:]
        prob, is_fault = predict_from_latency_window(latencies)
        return jsonify({
            "code": 200,
            "prob": float(f"{prob:.6f}"),
            "is_fault": is_fault,
            "threshold": THRESHOLD,
            "cluster": cluster_name,
            "mode": "latency_window",
            "model": MODEL_NAME,
        })
    except ValueError as exc:
        return jsonify({"code": 400, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"code": 500, "error": str(exc)}), 500


@app.route("/predict", methods=["POST"])
def predict_single_point():
    try:
        req = request.get_json(silent=True)
        if not isinstance(req, dict):
            return jsonify({"code": 400, "error": "Invalid JSON body"}), 400

        cluster_name = _get_cluster_name(req)
        if not cluster_name:
            return jsonify({"code": 400, "error": "Missing cluster name"}), 400

        value = req.get("latency", req.get("networkLatency"))
        if value is None:
            return jsonify({"code": 400, "error": "Missing field 'latency'"}), 400

        latency = float(value)
        if not np.isfinite(latency):
            return jsonify({"code": 400, "error": "latency is not finite"}), 400

        buffer = _get_buffer(cluster_name)
        with _buffers_lock:
            buffer.append(latency)
            latencies = list(buffer)

        if len(latencies) < LOOKBACK:
            return jsonify({
                "code": 202,
                "msg": f"Buffering... ({len(latencies)}/{LOOKBACK})",
                "prob": 0.0,
                "is_fault": False,
                "cluster": cluster_name,
                "mode": "single_point_buffer",
            }), 202

        prob, is_fault = predict_from_latency_window(latencies)
        return jsonify({
            "code": 200,
            "prob": float(f"{prob:.6f}"),
            "is_fault": is_fault,
            "threshold": THRESHOLD,
            "cluster": cluster_name,
            "mode": "single_point_buffer",
            "model": MODEL_NAME,
        })
    except ValueError as exc:
        return jsonify({"code": 400, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"code": 500, "error": str(exc)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "code": 200,
        "model": MODEL_NAME,
        "model_dir": str(MODEL_DIR),
        "lookback": LOOKBACK,
        "feature_names": FEATURE_NAMES,
        "threshold": THRESHOLD,
        "tau_net_ms": TAU_NET,
        "rolling_window": ROLLING_WINDOW,
        "window_endpoint": "/predict/window",
        "single_point_endpoint": "/predict",
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
