from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rote_mamba import RoteMamba


MODEL_NAMES = [
    "Raw-LSTM",
    "OPR-GRU-v2",
    "ROTE-GRU-v2",
    "ROTE-TCN-v2",
    "ROTE-Mamba-v2",
    "ROTE-Mamba-v2+Rank",
]

STATE_RISK_WEIGHT = {
    0: 1.0,   # stable
    1: 10.0,  # onset
    2: 1.5,   # persistent
    3: 2.0,   # recovery
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def load_npz(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def recarray_to_frame(meta: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame.from_records(meta)


@dataclass
class SplitIndex:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def chronological_split(n: int, train_frac: float = 0.70, val_frac: float = 0.10) -> SplitIndex:
    n_train = int(math.floor(n * train_frac))
    n_val = int(math.floor(n * val_frac))
    train = np.arange(0, n_train, dtype=np.int64)
    val = np.arange(n_train, n_train + n_val, dtype=np.int64)
    test = np.arange(n_train + n_val, n, dtype=np.int64)
    return SplitIndex(train=train, val=val, test=test)


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = x.reshape(-1, x.shape[-1]).mean(axis=0, keepdims=True)
        std = x.reshape(-1, x.shape[-1]).std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean.reshape(1, 1, -1)) / self.std.reshape(1, 1, -1)).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.reshape(-1).tolist(), "std": self.std.reshape(-1).tolist()}


class RoteDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        time_to_risk: np.ndarray,
        risk_intensity: np.ndarray,
        state_cls: np.ndarray,
        current_normal: np.ndarray,
        horizon: int,
    ) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.time = torch.from_numpy((time_to_risk.astype(np.float32) / float(horizon + 1)).reshape(-1, 1))
        self.intensity = torch.from_numpy(risk_intensity.astype(np.float32).reshape(-1, 1))
        self.state = torch.from_numpy(state_cls.astype(np.int64))
        self.current_normal = torch.from_numpy(current_normal.astype(np.float32))

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.x[idx], {
            "risk": self.y[idx],
            "time": self.time[idx],
            "intensity": self.intensity[idx],
            "state": self.state[idx],
            "current_normal": self.current_normal[idx],
        }


class RawLSTM(nn.Module):
    def __init__(self, input_dim: int = 1, hidden: int = 48, dropout: float = 0.15) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out, _ = self.rnn(x)
        return {"risk_logit": self.head(out[:, -1]).squeeze(-1)}


class OprGRU(nn.Module):
    def __init__(self, input_dim: int = 10, hidden: int = 64, dropout: float = 0.15) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out, _ = self.rnn(x)
        return {"risk_logit": self.head(out[:, -1]).squeeze(-1)}


class RoteGRU(nn.Module):
    def __init__(self, input_dim: int = 10, hidden: int = 64, dropout: float = 0.15) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.risk_head = nn.Linear(hidden, 1)
        self.time_head = nn.Linear(hidden, 1)
        self.intensity_head = nn.Linear(hidden, 1)
        self.state_head = nn.Linear(hidden, 4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out, _ = self.rnn(x)
        z = self.dropout(out[:, -1])
        return {
            "risk_logit": self.risk_head(z).squeeze(-1),
            "time_pred": torch.sigmoid(self.time_head(z)),
            "intensity_pred": torch.relu(self.intensity_head(z)),
            "state_logit": self.state_head(z),
        }


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + self.downsample(x))


class TcnBackbone(nn.Module):
    def __init__(self, input_dim: int, channels: tuple[int, ...] = (48, 64), kernel_size: int = 3, dropout: float = 0.15) -> None:
        super().__init__()
        blocks = []
        in_ch = input_dim
        for i, out_ch in enumerate(channels):
            blocks.append(TemporalBlock(in_ch, out_ch, kernel_size=kernel_size, dilation=2**i, dropout=dropout))
            in_ch = out_ch
        self.net = nn.Sequential(*blocks)
        self.out_dim = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        z = self.net(z)
        return z[:, :, -1]


class OprTCN(nn.Module):
    def __init__(self, input_dim: int = 10, dropout: float = 0.15) -> None:
        super().__init__()
        self.backbone = TcnBackbone(input_dim=input_dim, dropout=dropout)
        self.head = nn.Linear(self.backbone.out_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.backbone(x)
        return {"risk_logit": self.head(z).squeeze(-1)}


class ModernTCNBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 5, expansion: int = 2, dropout: float = 0.15) -> None:
        super().__init__()
        padding = kernel_size // 2
        hidden = int(d_model * expansion)
        self.norm = nn.LayerNorm(d_model)
        self.dwconv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding, groups=d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        z = self.dwconv(z.transpose(1, 2)).transpose(1, 2)
        return x + self.ffn(z)


class OprModernTCN(nn.Module):
    def __init__(self, input_dim: int = 13, d_model: int = 64, depth: int = 3, dropout: float = 0.15) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.blocks = nn.Sequential(*[ModernTCNBlock(d_model=d_model, kernel_size=5, dropout=dropout) for _ in range(depth)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.blocks(self.proj(x))
        h = self.pool(z.transpose(1, 2)).squeeze(-1)
        return {"risk_logit": self.head(h).squeeze(-1)}


class TSMixerBlock(nn.Module):
    def __init__(self, lookback: int, feature_dim: int, token_hidden: int = 32, feature_hidden: int = 64, dropout: float = 0.15) -> None:
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


class OprXLSTM(nn.Module):
    def __init__(self, input_dim: int = 13, hidden: int = 64, dropout: float = 0.15) -> None:
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden * 4)
        self.recurrent = nn.Linear(hidden, hidden * 4, bias=False)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = x.shape[0]
        hidden = self.recurrent.out_features // 4
        h = torch.zeros(batch, hidden, dtype=x.dtype, device=x.device)
        c = torch.zeros_like(h)
        for t in range(x.shape[1]):
            i, f, g, o = (self.in_proj(x[:, t]) + self.recurrent(h)).chunk(4, dim=-1)
            i = torch.exp(torch.clamp(i, max=6.0))
            f = torch.exp(torch.clamp(f, max=6.0))
            z = i + f + 1e-6
            i = i / z
            f = f / z
            c = f * c + i * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(self.norm(c))
        h = self.dropout(h)
        return {"risk_logit": self.head(h).squeeze(-1)}


class RoteTCN(nn.Module):
    def __init__(self, input_dim: int = 10, dropout: float = 0.15) -> None:
        super().__init__()
        self.backbone = TcnBackbone(input_dim=input_dim, dropout=dropout)
        d = self.backbone.out_dim
        self.risk_head = nn.Linear(d, 1)
        self.time_head = nn.Linear(d, 1)
        self.intensity_head = nn.Linear(d, 1)
        self.state_head = nn.Linear(d, 4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.backbone(x)
        return {
            "risk_logit": self.risk_head(z).squeeze(-1),
            "time_pred": torch.sigmoid(self.time_head(z)),
            "intensity_pred": torch.relu(self.intensity_head(z)),
            "state_logit": self.state_head(z),
        }


def normalize_model_name(name: str) -> str:
    name = (
        str(name)
        .replace("-onsetAware", "")
        .replace("-v2", "")
        .replace("+Rank", "")
    )
    if name in {"TSMixer-Raw", "TSMixer-BasicStat", "TSMixer-OPR"}:
        return "OPR-TSMixer"
    if name.startswith("TSMixer-OPRGroup-"):
        return "OPR-TSMixer"
    return name


def has_ranking_loss(name: str) -> bool:
    return "+Rank" in str(name)


def make_model(name: str, input_dim: int) -> nn.Module:
    base_name = normalize_model_name(name)
    if base_name == "Raw-LSTM":
        return RawLSTM(input_dim=input_dim)
    if base_name == "OPR-GRU":
        return OprGRU(input_dim=input_dim)
    if base_name == "OPR-TCN":
        return OprTCN(input_dim=input_dim)
    if base_name == "OPR-ModernTCN":
        return OprModernTCN(input_dim=input_dim)
    if base_name == "OPR-TSMixer":
        return OprTSMixer(input_dim=input_dim)
    if base_name == "OPR-xLSTM":
        return OprXLSTM(input_dim=input_dim)
    if base_name == "ROTE-GRU":
        return RoteGRU(input_dim=input_dim)
    if base_name == "ROTE-TCN":
        return RoteTCN(input_dim=input_dim)
    if base_name == "ROTE-Mamba":
        return RoteMamba(input_dim=input_dim)
    raise ValueError(f"Unknown model: {name}")


def binary_pos_weight(y: np.ndarray) -> torch.Tensor:
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    if pos <= 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(neg / pos, dtype=torch.float32)


def state_class_weight(state: np.ndarray) -> torch.Tensor:
    counts = np.bincount(state.astype(int), minlength=4).astype(np.float64)
    total = counts.sum()
    weights = total / (4.0 * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def scheduler_aware_ranking_loss(out: dict[str, torch.Tensor], batch_y: dict[str, torch.Tensor]) -> torch.Tensor:
    cn_mask = batch_y["current_normal"] > 0.5
    pos_mask = cn_mask & (batch_y["risk"] > 0.5)
    neg_mask = cn_mask & (batch_y["risk"] <= 0.5)
    if not torch.any(pos_mask) or not torch.any(neg_mask):
        return torch.zeros((), dtype=out["risk_logit"].dtype, device=out["risk_logit"].device)

    score = out["risk_logit"]
    if "intensity_pred" in out:
        score = score + 0.5 * out["intensity_pred"].squeeze(-1)
    if "time_pred" in out:
        score = score - 0.3 * out["time_pred"].squeeze(-1)

    pairwise_margin = score[pos_mask][:, None] - score[neg_mask][None, :]
    return torch.nn.functional.softplus(-pairwise_margin).mean()


def compute_loss(
    model_name: str,
    out: dict[str, torch.Tensor],
    batch_y: dict[str, torch.Tensor],
    bce: nn.Module,
    time_loss_fn: nn.Module,
    intensity_loss_fn: nn.Module,
    state_loss_fn: nn.Module,
) -> torch.Tensor:
    raw_risk_loss = nn.functional.binary_cross_entropy_with_logits(
        out["risk_logit"],
        batch_y["risk"],
        reduction="none",
    )
    state_weight = torch.ones_like(raw_risk_loss)
    for state_id, weight in STATE_RISK_WEIGHT.items():
        state_weight = torch.where(
            batch_y["state"] == int(state_id),
            torch.full_like(state_weight, float(weight)),
            state_weight,
        )
    risk_loss_all = (raw_risk_loss * state_weight).mean()

    cn_mask = batch_y["current_normal"] > 0.5
    if torch.any(cn_mask):
        risk_loss_current_normal = raw_risk_loss[cn_mask].mean()
    else:
        risk_loss_current_normal = torch.zeros((), device=raw_risk_loss.device)

    risk_loss = risk_loss_all + 2.0 * risk_loss_current_normal
    if not normalize_model_name(model_name).startswith("ROTE-"):
        return risk_loss

    time_loss = time_loss_fn(out["time_pred"], batch_y["time"])
    intensity_loss = intensity_loss_fn(out["intensity_pred"], batch_y["intensity"])
    state_loss = state_loss_fn(out["state_logit"], batch_y["state"])
    total = risk_loss + 0.3 * time_loss + 0.2 * intensity_loss + 0.5 * state_loss
    if has_ranking_loss(model_name):
        total = total + 0.5 * scheduler_aware_ranking_loss(out, batch_y)
    return total


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    logits = []
    for x, _ in loader:
        x = x.to(device)
        out = model(x)
        logits.append(out["risk_logit"].detach().cpu().numpy())
    return np.concatenate(logits, axis=0)


def best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        return 0.5, float("nan")
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    if len(thresholds) == 0:
        return 0.5, float(np.nanmax(f1))
    best_idx = int(np.nanargmax(f1[:-1]))
    return float(thresholds[best_idx]), float(f1[best_idx])


def precision_at_k(y_true: np.ndarray, prob: np.ndarray, k: int) -> float:
    if len(y_true) == 0:
        return float("nan")
    k = min(int(k), len(y_true))
    if k <= 0:
        return float("nan")
    idx = np.argsort(-prob)[:k]
    return float(np.mean(y_true[idx]))


def metrics_for_subset(
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    name: str,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= float(threshold)).astype(int)
    has_two_classes = len(np.unique(y_true)) == 2
    n_pos = int(np.sum(y_true == 1))

    return {
        "subset": name,
        "n": int(len(y_true)),
        "positives": n_pos,
        "positive_rate": float(n_pos / max(1, len(y_true))),
        "roc_auc": float(roc_auc_score(y_true, prob)) if has_two_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, prob)) if has_two_classes else float("nan"),
        "brier": float(brier_score_loss(y_true, prob)) if len(y_true) else float("nan"),
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, pred, zero_division=0)) if len(y_true) else float("nan"),
        "precision": float(precision_score(y_true, pred, zero_division=0)) if len(y_true) else float("nan"),
        "recall": float(recall_score(y_true, pred, zero_division=0)) if len(y_true) else float("nan"),
        "p_at_100": precision_at_k(y_true, prob, 100),
        "p_at_500": precision_at_k(y_true, prob, 500),
        "p_at_1000": precision_at_k(y_true, prob, 1000),
        "p_at_pos": precision_at_k(y_true, prob, max(1, n_pos)),
        "mean_prob": float(np.mean(prob)) if len(prob) else float("nan"),
        "mean_prob_pos": float(np.mean(prob[y_true == 1])) if n_pos else float("nan"),
        "mean_prob_neg": float(np.mean(prob[y_true == 0])) if n_pos < len(y_true) else float("nan"),
    }


def pr_auc_or_nan(y_true: np.ndarray, prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, prob))


@torch.no_grad()
def validation_selection_score(
    model: nn.Module,
    loader: DataLoader,
    y_val: np.ndarray,
    current_normal_val: np.ndarray,
    device: torch.device,
) -> dict:
    logits = predict_logits(model, loader, device=device)
    prob = sigmoid_np(logits)
    val_all_pr_auc = pr_auc_or_nan(y_val, prob)
    mask = np.asarray(current_normal_val, dtype=int) == 1
    val_current_normal_pr_auc = pr_auc_or_nan(y_val[mask], prob[mask])

    if np.isfinite(val_current_normal_pr_auc) and np.isfinite(val_all_pr_auc):
        selection_score = 0.7 * val_current_normal_pr_auc + 0.3 * val_all_pr_auc
    elif np.isfinite(val_current_normal_pr_auc):
        selection_score = val_current_normal_pr_auc
    elif np.isfinite(val_all_pr_auc):
        selection_score = val_all_pr_auc
    else:
        selection_score = float("-inf")

    return {
        "selection_score": float(selection_score),
        "val_current_normal_pr_auc": float(val_current_normal_pr_auc),
        "val_all_pr_auc": float(val_all_pr_auc),
        "prob": prob,
    }


def train_one_model(
    model_name: str,
    x_all: np.ndarray,
    labels: dict[str, np.ndarray],
    split: SplitIndex,
    horizon: int,
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
) -> tuple[nn.Module, Standardizer, float, dict]:
    raw_only = model_name.startswith("Raw-LSTM")
    x = x_all[:, :, 0:1] if raw_only else x_all
    scaler = Standardizer.fit(x[split.train])
    x_scaled = scaler.transform(x)

    train_ds = RoteDataset(
        x_scaled[split.train],
        labels["y_risk"][split.train],
        labels["time_to_risk"][split.train],
        labels["risk_intensity"][split.train],
        labels["state_cls"][split.train],
        labels["current_normal"][split.train],
        horizon=horizon,
    )
    val_ds = RoteDataset(
        x_scaled[split.val],
        labels["y_risk"][split.val],
        labels["time_to_risk"][split.val],
        labels["risk_intensity"][split.val],
        labels["state_cls"][split.val],
        labels["current_normal"][split.val],
        horizon=horizon,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = make_model(model_name, input_dim=x.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    pos_weight = binary_pos_weight(labels["y_risk"][split.train]).to(device)
    state_weight = state_class_weight(labels["state_cls"][split.train]).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    time_loss_fn = nn.SmoothL1Loss()
    intensity_loss_fn = nn.SmoothL1Loss()
    state_loss_fn = nn.CrossEntropyLoss(weight=state_weight)

    best_score = float("-inf")
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    best_selection: dict[str, float] = {}
    patience_left = int(args.patience)
    history = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = {k: v.to(device) for k, v in batch_y.items()}
            opt.zero_grad(set_to_none=True)
            out = model(batch_x)
            loss = compute_loss(
                model_name,
                out,
                batch_y,
                bce=bce,
                time_loss_fn=time_loss_fn,
                intensity_loss_fn=intensity_loss_fn,
                state_loss_fn=state_loss_fn,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = {k: v.to(device) for k, v in batch_y.items()}
                out = model(batch_x)
                loss = compute_loss(
                    model_name,
                    out,
                    batch_y,
                    bce=bce,
                    time_loss_fn=time_loss_fn,
                    intensity_loss_fn=intensity_loss_fn,
                    state_loss_fn=state_loss_fn,
                )
                val_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        selection = validation_selection_score(
            model,
            val_loader,
            labels["y_risk"][split.val],
            labels["current_normal"][split.val],
            device=device,
        )
        score = float(selection["selection_score"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "selection_score": score,
                "val_current_normal_pr_auc": float(selection["val_current_normal_pr_auc"]),
                "val_all_pr_auc": float(selection["val_all_pr_auc"]),
            }
        )
        print(
            f"{model_name} epoch {epoch:03d}: train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} val_cn_pr={selection['val_current_normal_pr_auc']:.5f} "
            f"val_all_pr={selection['val_all_pr_auc']:.5f} score={score:.5f}"
        )

        if score > best_score + 1e-5:
            best_score = score
            best_val = val_loss
            best_epoch = epoch
            best_selection = {k: float(v) for k, v in selection.items() if k != "prob"}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(args.patience)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_logits = predict_logits(model, val_loader, device=device)
    val_prob = sigmoid_np(val_logits)
    threshold, val_best_f1 = best_f1_threshold(labels["y_risk"][split.val], val_prob)

    model_dir = out_dir / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pt")
    (model_dir / "scaler.json").write_text(json.dumps(scaler.to_dict(), indent=2), encoding="utf-8")
    meta = {
        "model_name": model_name,
        "input_dim": int(x.shape[-1]),
        "raw_only": bool(raw_only),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "best_selection_score": float(best_score),
        "best_selection": best_selection,
        "val_threshold_best_f1": float(threshold),
        "val_best_f1": float(val_best_f1),
        "pos_weight": float(pos_weight.detach().cpu()),
        "history": history,
    }
    (model_dir / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return model, scaler, threshold, meta


def eval_model_on_array(
    model: nn.Module,
    scaler: Standardizer,
    threshold: float,
    model_name: str,
    x_all: np.ndarray,
    y: np.ndarray,
    horizon: int,
    device: torch.device,
    args: argparse.Namespace,
    time_to_risk: np.ndarray,
    risk_intensity: np.ndarray,
    state_cls: np.ndarray,
    current_normal: np.ndarray,
) -> np.ndarray:
    raw_only = model_name.startswith("Raw-LSTM")
    x = x_all[:, :, 0:1] if raw_only else x_all
    x = scaler.transform(x)
    ds = RoteDataset(x, y, time_to_risk, risk_intensity, state_cls, current_normal, horizon=horizon)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    logits = predict_logits(model, loader, device=device)
    return sigmoid_np(logits)


def evaluate_all(
    model_name: str,
    model: nn.Module,
    scaler: Standardizer,
    threshold: float,
    real: dict,
    holdout: dict,
    split: SplitIndex,
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
) -> list[dict]:
    horizon = int(real["horizon"])
    real_meta = recarray_to_frame(real["meta"])
    holdout_meta = recarray_to_frame(holdout["meta"])

    real_test_idx = split.test
    prob_real_test = eval_model_on_array(
        model,
        scaler,
        threshold,
        model_name,
        real["X_opr"][real_test_idx],
        real["y_risk"][real_test_idx],
        horizon,
        device,
        args,
        real["time_to_risk"][real_test_idx],
        real["risk_intensity"][real_test_idx],
        real["state_cls"][real_test_idx],
        real["current_normal"][real_test_idx],
    )

    prob_holdout = eval_model_on_array(
        model,
        scaler,
        threshold,
        model_name,
        holdout["X_opr"],
        holdout["y_risk"],
        int(holdout["horizon"]),
        device,
        args,
        holdout["time_to_risk"],
        holdout["risk_intensity"],
        holdout["state_cls"],
        holdout["current_normal"],
    )

    rows = []
    y_real_test = real["y_risk"][real_test_idx]
    meta_real_test = real_meta.iloc[real_test_idx].reset_index(drop=True)
    rows.append(metrics_for_subset(y_real_test, prob_real_test, threshold, "real_test_all"))

    mask_real_cn = meta_real_test["current_normal"].astype(int).to_numpy() == 1
    rows.append(metrics_for_subset(y_real_test[mask_real_cn], prob_real_test[mask_real_cn], threshold, "real_test_current_normal"))

    y_holdout = holdout["y_risk"].astype(int)
    mask_holdout_cn = holdout_meta["current_normal"].astype(int).to_numpy() == 1
    rows.append(metrics_for_subset(y_holdout[mask_holdout_cn], prob_holdout[mask_holdout_cn], threshold, "holdout_current_normal"))

    future_type = holdout_meta["future_onset_type"].astype(str)
    mask_predictable = mask_holdout_cn & (future_type.to_numpy() == "predictable-onset")
    rows.append(metrics_for_subset(y_holdout[mask_predictable], prob_holdout[mask_predictable], threshold, "holdout_predictable_onset"))

    mask_abrupt = mask_holdout_cn & (future_type.to_numpy() == "abrupt-onset")
    rows.append(metrics_for_subset(y_holdout[mask_abrupt], prob_holdout[mask_abrupt], threshold, "holdout_abrupt_onset"))

    for row in rows:
        row["model"] = model_name

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "global_idx": real_test_idx,
            "prob": prob_real_test,
            "y_risk": y_real_test,
            "current_normal": meta_real_test["current_normal"].astype(int).to_numpy(),
            "state_cls": meta_real_test["state_cls"].astype(int).to_numpy(),
            "state_name": meta_real_test["state_name"].astype(str).to_numpy(),
        }
    ).to_csv(pred_dir / f"{model_name}_real_test_predictions.csv", index=False)

    pd.DataFrame(
        {
            "prob": prob_holdout,
            "y_risk": y_holdout,
            "current_normal": holdout_meta["current_normal"].astype(int).to_numpy(),
            "state_cls": holdout_meta["state_cls"].astype(int).to_numpy(),
            "state_name": holdout_meta["state_name"].astype(str).to_numpy(),
            "future_onset_type": holdout_meta["future_onset_type"].astype(str).to_numpy(),
            "block_type": holdout_meta["block_type"].astype(str).to_numpy(),
            "phase": holdout_meta["phase"].astype(str).to_numpy(),
        }
    ).to_csv(pred_dir / f"{model_name}_holdout_predictions.csv", index=False)

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Train second-round onset-aware ROTE experiments.")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(int(args.seed))
    root = args.root.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else root / "rote_training_outputs_round2"
    out_dir.mkdir(parents=True, exist_ok=True)

    real = load_npz(root / "rote_dataset_real.npz")
    holdout = load_npz(root / "rote_dataset_holdout_v2.npz")
    split = chronological_split(len(real["y_risk"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = {
        "y_risk": real["y_risk"].astype(np.int64),
        "time_to_risk": real["time_to_risk"].astype(np.int64),
        "risk_intensity": real["risk_intensity"].astype(np.float32),
        "state_cls": real["state_cls"].astype(np.int64),
        "current_normal": real["current_normal"].astype(np.int64),
    }

    config = {
        "real_npz": str(root / "rote_dataset_real.npz"),
        "holdout_npz": str(root / "rote_dataset_holdout_v2.npz"),
        "split": {
            "train": [int(split.train[0]), int(split.train[-1]), int(len(split.train))],
            "val": [int(split.val[0]), int(split.val[-1]), int(len(split.val))],
            "test": [int(split.test[0]), int(split.test[-1]), int(len(split.test))],
        },
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
            real["X_opr"].astype(np.float32),
            labels,
            split,
            horizon=int(real["horizon"]),
            args=args,
            out_dir=out_dir,
            device=device,
        )
        train_meta[model_name] = meta
        rows = evaluate_all(
            model_name,
            model,
            scaler,
            threshold,
            real,
            holdout,
            split,
            args,
            out_dir,
            device,
        )
        all_metrics.extend(rows)

    metrics_df = pd.DataFrame(all_metrics)
    cols = ["model", "subset"] + [c for c in metrics_df.columns if c not in {"model", "subset"}]
    metrics_df = metrics_df[cols]
    metrics_df.to_csv(out_dir / "metrics_summary.csv", index=False)
    (out_dir / "metrics_summary.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "train_meta_all.json").write_text(json.dumps(train_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== Metrics Summary =====")
    view = metrics_df[metrics_df["subset"].isin(["real_test_all", "real_test_current_normal", "holdout_current_normal"])]
    print(view[["model", "subset", "n", "positives", "roc_auc", "pr_auc", "brier", "f1", "precision", "recall", "p_at_100"]].to_string(index=False))
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
