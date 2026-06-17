from __future__ import annotations

import torch
from torch import nn


class OnsetAwareTokenGate(nn.Module):
    """Content gate that emphasizes tokens with onset-like OPR evidence."""

    def __init__(self, d_model: int, feature_dim: int = 10, dropout: float = 0.1) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.gate = nn.Sequential(
            nn.Linear(d_model + 4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        if x_raw.shape[-1] >= 10:
            onset_cues = torch.stack(
                [
                    x_raw[..., 1],  # delta_latency
                    x_raw[..., 7],  # near_threshold_ratio
                    x_raw[..., 8],  # pos_delta_latency
                    x_raw[..., 9],  # pos_delta_std
                ],
                dim=-1,
            )
        else:
            onset_cues = torch.zeros(*x_raw.shape[:2], 4, dtype=x_raw.dtype, device=x_raw.device)

        gate = self.gate(torch.cat([z, onset_cues], dim=-1))
        return z * (0.5 + gate)


class MambaOrFallbackBlock(nn.Module):
    """Use official mamba-ssm when available; otherwise use a selective SSM-style fallback."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.uses_mamba_ssm = False

        try:
            from mamba_ssm import Mamba  # type: ignore

            self.mixer = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.uses_mamba_ssm = True
        except Exception:
            hidden = int(d_model * expand)
            self.in_proj = nn.Linear(d_model, hidden * 3)
            self.state_proj = nn.Linear(hidden, d_model)
            self.out_proj = nn.Linear(d_model, d_model)

    def _fallback_forward(self, x: torch.Tensor) -> torch.Tensor:
        u, delta, gate = self.in_proj(x).chunk(3, dim=-1)
        u = torch.tanh(u)
        decay = torch.sigmoid(delta)
        gate = torch.sigmoid(gate)

        h = torch.zeros(x.shape[0], u.shape[-1], dtype=x.dtype, device=x.device)
        states = []
        for t in range(x.shape[1]):
            h = decay[:, t] * h + (1.0 - decay[:, t]) * u[:, t]
            states.append(h)
        y = torch.stack(states, dim=1) * gate
        y = self.state_proj(y)
        return self.out_proj(torch.nn.functional.gelu(y))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        if self.uses_mamba_ssm:
            y = self.mixer(z)
        else:
            y = self._fallback_forward(z)
        return x + self.dropout(y)


class TemporalAttentionPooling(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(z).squeeze(-1), dim=-1)
        return torch.sum(z * weights.unsqueeze(-1), dim=1)


class RoteMamba(nn.Module):
    """ROTE-Mamba: OPR tokens, onset-aware gate, selective SSM encoder, multi-task heads."""

    def __init__(
        self,
        input_dim: int = 10,
        d_model: int = 64,
        n_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.token_gate = OnsetAwareTokenGate(d_model=d_model, feature_dim=input_dim, dropout=dropout)
        self.encoder = nn.ModuleList(
            [
                MambaOrFallbackBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(int(n_layers))
            ]
        )
        self.pool = TemporalAttentionPooling(d_model=d_model)
        self.head_norm = nn.LayerNorm(d_model)

        self.risk_head = nn.Linear(d_model, 1)
        self.time_head = nn.Linear(d_model, 1)
        self.intensity_head = nn.Linear(d_model, 1)
        self.state_head = nn.Linear(d_model, 4)

    @property
    def uses_mamba_ssm(self) -> bool:
        return any(getattr(layer, "uses_mamba_ssm", False) for layer in self.encoder)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.input_proj(x)
        z = self.token_gate(z, x)
        for layer in self.encoder:
            z = layer(z)
        h = self.head_norm(self.pool(z))
        return {
            "risk_logit": self.risk_head(h).squeeze(-1),
            "time_pred": torch.sigmoid(self.time_head(h)),
            "intensity_pred": torch.relu(self.intensity_head(h)),
            "state_logit": self.state_head(h),
        }
